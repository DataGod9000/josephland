import { dialogueData, scaleFactor } from "./constants";
import { k } from "./kaboomCtx";
import { displayDialogue, setCamScale } from "./utils";

k.loadSprite("spritesheet", "./spritesheet.png", {
  sliceX: 39,
  sliceY: 31,
  anims: {
    "idle-down": 936,
    "walk-down": { from: 936, to: 939, loop: true, speed: 8 },
    "idle-side": 975,
    "walk-side": { from: 975, to: 978, loop: true, speed: 8 },
    "idle-up": 1014,
    "walk-up": { from: 1014, to: 1017, loop: true, speed: 8 },
  },
});

k.loadSprite("island_map", "./joseph_island.png");
k.loadSprite("room_map", "./map.png");
k.loadSound("bgm", "./background_music.mp3");

k.setBackground(k.Color.fromHex("#311047"));

function addPlayer(map, scaleFactor) {
  return k.make([
    k.sprite("spritesheet", { anim: "idle-down" }),
    k.area({
      shape: new k.Rect(k.vec2(0, 3), 10, 10),
    }),
    k.body(),
    k.anchor("center"),
    k.pos(),
    k.scale(scaleFactor),
    {
      speed: 125,
      direction: "down",
      isInDialogue: false,
    },
    "player",
  ]);
}

function setupScene(mapData, mapSpriteName, scaleFactor, onEnterhouse = null, onExithouse = null, useExitSpawn = false) {
  const layers = mapData.layers;
  const map = k.add([k.sprite(mapSpriteName), k.pos(0), k.scale(scaleFactor)]);
  const player = addPlayer(map, scaleFactor);

  for (const layer of layers) {
    if (layer.name === "boundaries") {
      for (const boundary of layer.objects) {
        if (boundary.width <= 0 || boundary.height <= 0) continue;
        map.add([
          k.area({
            shape: new k.Rect(k.vec2(0), boundary.width, boundary.height),
          }),
          k.body({ isStatic: true }),
          k.pos(boundary.x, boundary.y),
          boundary.name,
        ]);

        if (boundary.name) {
          player.onCollide(boundary.name, () => {
            player.isInDialogue = true;
            displayDialogue(
              dialogueData[boundary.name],
              () => (player.isInDialogue = false)
            );
          });
        }
      }
      continue;
    }

    if (layer.name === "enterhouse" && onEnterhouse) {
      for (const obj of layer.objects) {
        const w = obj.width || 24;
        const h = obj.height || 24;
        map.add([
          k.area({ shape: new k.Rect(k.vec2(0), w, h) }),
          k.pos(obj.x, obj.y),
          "enterhouse",
        ]);
      }
      player.onCollide("enterhouse", onEnterhouse);
      continue;
    }

    if (layer.name === "exithouse" && onExithouse) {
      for (const obj of layer.objects) {
        const w = obj.width || 24;
        const h = obj.height || 24;
        map.add([
          k.area({ shape: new k.Rect(k.vec2(0), w, h) }),
          k.pos(obj.x, obj.y),
          "exithouse",
        ]);
      }
      player.onCollide("exithouse", onExithouse);
      continue;
    }

    if (layer.name === "spawnpoints") {
      const spawnName = useExitSpawn ? "player_exit" : "player";
      let placed = false;
      for (const entity of layer.objects) {
        if (entity.name === spawnName) {
          player.pos = k.vec2(
            (map.pos.x + entity.x) * scaleFactor,
            (map.pos.y + entity.y) * scaleFactor
          );
          placed = true;
          break;
        }
      }
      if (!placed) {
        const fallback = layer.objects.find((e) => e.name === "player");
        if (fallback) {
          player.pos = k.vec2(
            (map.pos.x + fallback.x) * scaleFactor,
            (map.pos.y + fallback.y) * scaleFactor
          );
        }
      }
      k.add(player);
    }
  }

  return { map, player };
}

function addMovementAndCamera(player) {
  k.onUpdate(() => {
    k.camPos(player.worldPos().x, player.worldPos().y - 100);
  });

  k.onMouseDown((mouseBtn) => {
    if (mouseBtn !== "left" || player.isInDialogue) return;

    const worldMousePos = k.toWorld(k.mousePos());
    player.moveTo(worldMousePos, player.speed);

    const mouseAngle = player.pos.angle(worldMousePos);

    const lowerBound = 50;
    const upperBound = 125;

    if (
      mouseAngle > lowerBound &&
      mouseAngle < upperBound &&
      player.curAnim() !== "walk-up"
    ) {
      player.play("walk-up");
      player.direction = "up";
      return;
    }

    if (
      mouseAngle < -lowerBound &&
      mouseAngle > -upperBound &&
      player.curAnim() !== "walk-down"
    ) {
      player.play("walk-down");
      player.direction = "down";
      return;
    }

    if (Math.abs(mouseAngle) > upperBound) {
      player.flipX = false;
      if (player.curAnim() !== "walk-side") player.play("walk-side");
      player.direction = "right";
      return;
    }

    if (Math.abs(mouseAngle) < lowerBound) {
      player.flipX = true;
      if (player.curAnim() !== "walk-side") player.play("walk-side");
      player.direction = "left";
      return;
    }
  });

  function stopAnims() {
    if (player.direction === "down") {
      player.play("idle-down");
      return;
    }
    if (player.direction === "up") {
      player.play("idle-up");
      return;
    }
    player.play("idle-side");
  }

  k.onMouseRelease(stopAnims);
  k.onKeyRelease(stopAnims);

  k.onKeyDown((key) => {
    const keyMap = [
      k.isKeyDown("right"),
      k.isKeyDown("left"),
      k.isKeyDown("up"),
      k.isKeyDown("down"),
    ];
    let nbOfKeyPressed = 0;
    for (const key of keyMap) {
      if (key) nbOfKeyPressed++;
    }
    if (nbOfKeyPressed > 1) return;
    if (player.isInDialogue) return;
    if (keyMap[0]) {
      player.flipX = false;
      if (player.curAnim() !== "walk-side") player.play("walk-side");
      player.direction = "right";
      player.move(player.speed, 0);
      return;
    }
    if (keyMap[1]) {
      player.flipX = true;
      if (player.curAnim() !== "walk-side") player.play("walk-side");
      player.direction = "left";
      player.move(-player.speed, 0);
      return;
    }
    if (keyMap[2]) {
      if (player.curAnim() !== "walk-up") player.play("walk-up");
      player.direction = "up";
      player.move(0, -player.speed);
      return;
    }
    if (keyMap[3]) {
      if (player.curAnim() !== "walk-down") player.play("walk-down");
      player.direction = "down";
      player.move(0, player.speed);
    }
  });
}

k.scene("island", async (opts = {}) => {
  const loadingEl = document.getElementById("loading-screen");
  if (loadingEl) loadingEl.style.display = "none";
  if (!window.__bgmStarted) {
    window.__bgmStarted = true;
    k.play("bgm", { loop: true, volume: 0.4 });
  }
  const useExitSpawn = opts.useExitSpawn === true;
  const mapData = await (await fetch("./joseph_island.json")).json();
  const { player } = setupScene(
    mapData,
    "island_map",
    scaleFactor,
    () => k.go("room"),
    null,
    useExitSpawn
  );

  setCamScale(k);
  k.onResize(() => setCamScale(k));
  addMovementAndCamera(player);
});

k.scene("room", async () => {
  const mapData = await (await fetch("./map.json")).json();
  const { player } = setupScene(
    mapData,
    "room_map",
    scaleFactor,
    null,
    () => k.go("island", { useExitSpawn: true })
  );

  setCamScale(k);
  k.onResize(() => setCamScale(k));
  addMovementAndCamera(player);
});

k.go("island");
