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
    // 2-frame “bob” so NPC looks like it’s moving (adjust from/to to match your sheet)
    "npc-slime": { from: 858, to: 859, loop: true, speed: 4 },
    "npc-kiki": { from: 780, to: 781, loop: true, speed: 4 },
    "npc-moca": { from: 788, to: 789, loop: true, speed: 4 },
  },
});

k.loadSprite("island_map", "./joseph_island.png");
k.loadSprite("room_map", "./map.png");
k.loadSound("bgm", "./background_music.mp3");
k.loadSound("grass_walk", "./grass_walking.m4a");
k.loadSound("door_open", "./door_open.m4a");
k.loadSound("floor_walk", "./floor_walk.m4a");

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
    k.z(1),
    {
      speed: 125,
      direction: "down",
      isInDialogue: false,
    },
    "player",
  ]);
}

function setupScene(mapData, mapSpriteName, scaleFactor, onEnterhouse = null, onExithouse = null, useExitSpawn = false, dialogueCooldownSeconds = 0) {
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
            player.vel = k.vec2(0, 0);
            if (player.direction === "down") player.play("idle-down");
            else if (player.direction === "up") player.play("idle-up");
            else player.play("idle-side");
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

    if (layer.name === "dialogue_triggers") {
      if (dialogueCooldownSeconds > 0 && !window.__dialogueCooldowns) window.__dialogueCooldowns = {};
      for (const obj of layer.objects) {
        if (!obj.name || !dialogueData[obj.name]) continue;
        const w = Math.max(1, obj.width || 16);
        const h = Math.max(1, obj.height || 16);
        map.add([
          k.area({ shape: new k.Rect(k.vec2(0), w, h) }),
          k.pos(obj.x, obj.y),
          obj.name,
        ]);
        const triggerName = obj.name;
        const triggerX = obj.x;
        const triggerY = obj.y;
        const triggerW = w;
        const triggerH = h;
        const noEngageFromTop = triggerName === "Janet" || triggerName === "wenzheng" || triggerName === "mom" || triggerName === "dad" || triggerName === "shelves";
        player.onCollide(triggerName, () => {
          if (player.isInDialogue) return;
          if (dialogueCooldownSeconds > 0) {
            const now = k.time();
            if (now - (window.__dialogueCooldowns[triggerName] || 0) < dialogueCooldownSeconds) return;
          }
          if (noEngageFromTop) {
            const triggerCenterYWorld = (map.pos.y + (triggerY + triggerH * 0.5)) * scaleFactor;
            const triggerCenterXWorld = (map.pos.x + (triggerX + triggerW * 0.5)) * scaleFactor;
            const fromFront = player.pos.y >= triggerCenterYWorld;
            if (triggerName === "Janet" || triggerName === "dad") {
              const fromRight = player.pos.x >= triggerCenterXWorld;
              if (!fromFront && !fromRight) return;
            } else if (triggerName === "mom") {
              const fromLeft = player.pos.x <= triggerCenterXWorld;
              if (!fromFront && !fromLeft) return;
            } else {
              if (!fromFront) return;
            }
          }
          player.vel = k.vec2(0, 0);
          if (player.direction === "down") player.play("idle-down");
          else if (player.direction === "up") player.play("idle-up");
          else player.play("idle-side");
          player.isInDialogue = true;
          displayDialogue(dialogueData[triggerName], () => {
            player.isInDialogue = false;
            if (dialogueCooldownSeconds > 0) window.__dialogueCooldowns[triggerName] = k.time();
          });
        });
      }
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

const STEP_INTERVAL = 0.35;

function addMovementAndCamera(player, options = {}) {
  const onStep = options.onStep;
  let isMoving = false;
  let lastStepTime = 0;

  k.onUpdate(() => {
    if (player.isInDialogue) {
      player.vel = k.vec2(0, 0);
      return;
    }
    k.camPos(player.worldPos().x, player.worldPos().y - 100);
    if (onStep && isMoving) {
      const now = k.time();
      if (lastStepTime === 0 || now - lastStepTime >= STEP_INTERVAL) {
        onStep();
        lastStepTime = now;
      }
    }
  });

  k.onMouseDown((mouseBtn) => {
    if (mouseBtn !== "left" || player.isInDialogue) return;
    isMoving = true;
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
    isMoving = false;
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
      isMoving = true;
      player.flipX = false;
      if (player.curAnim() !== "walk-side") player.play("walk-side");
      player.direction = "right";
      player.move(player.speed, 0);
      return;
    }
    if (keyMap[1]) {
      isMoving = true;
      player.flipX = true;
      if (player.curAnim() !== "walk-side") player.play("walk-side");
      player.direction = "left";
      player.move(-player.speed, 0);
      return;
    }
    if (keyMap[2]) {
      isMoving = true;
      if (player.curAnim() !== "walk-up") player.play("walk-up");
      player.direction = "up";
      player.move(0, -player.speed);
      return;
    }
    if (keyMap[3]) {
      isMoving = true;
      if (player.curAnim() !== "walk-down") player.play("walk-down");
      player.direction = "down";
      player.move(0, player.speed);
    }
  });
}

k.scene("island", async (opts = {}) => {
  const loadingEl = document.getElementById("loading-screen");
  const tapEl = document.getElementById("tap-to-start");
  if (tapEl) tapEl.style.display = "none";

  const useExitSpawn = opts.useExitSpawn === true;
  const mapData = await (await fetch("./joseph_island.json")).json();
  const { player } = setupScene(
    mapData,
    "island_map",
    scaleFactor,
    () => {
      k.play("door_open", { volume: 1 });
      k.go("room");
    },
    null,
    useExitSpawn,
    2
  );

  // NPCs: same spritesheet, 2-frame anim so they look like they’re moving
  const npcAnims = {
    into_slime: "npc-slime",
  };
  const triggersLayer = mapData.layers?.find((l) => l.name === "dialogue_triggers");
  if (triggersLayer) {
    for (const obj of triggersLayer.objects) {
      const anim = npcAnims[obj.name];
      if (!anim) continue;
      const x = (obj.x + (obj.width || 0) / 2) * scaleFactor;
      const y = (obj.y + (obj.height || 0) / 2) * scaleFactor;
      k.add([
        k.sprite("spritesheet", { anim }),
        k.pos(x, y),
        k.anchor("center"),
        k.scale(scaleFactor),
        k.z(0),
      ]);
    }
  }

  setCamScale(k);
  k.onResize(() => setCamScale(k));
  addMovementAndCamera(player, {
    onStep: () => {
      k.play("grass_walk", { volume: 0.7 });
    },
  });

  if (loadingEl) loadingEl.style.display = "none";
  if (!window.__bgmStarted && tapEl) {
    tapEl.style.display = "flex";
    const startBgm = () => {
      window.__bgmStarted = true;
      k.play("bgm", { loop: true, volume: 0.48 });
      tapEl.style.display = "none";
      document.removeEventListener("mousedown", startBgm);
      document.removeEventListener("touchstart", startBgm);
      document.removeEventListener("keydown", startBgm);
    };
    document.addEventListener("mousedown", startBgm, { once: true });
    document.addEventListener("touchstart", startBgm, { once: true });
    document.addEventListener("keydown", startBgm, { once: true });
  } else if (window.__bgmStarted && tapEl) {
    tapEl.style.display = "none";
  }
});

k.scene("room", async () => {
  const mapData = await (await fetch("./map.json")).json();
  const { player } = setupScene(
    mapData,
    "room_map",
    scaleFactor,
    null,
    () => {
      k.play("door_open", { volume: 1 });
      k.go("island", { useExitSpawn: true });
    },
    null,
    false,
    0.4
  );

  // Kiki and Moca sprites (2-frame bob) at their trigger positions
  const roomNpcAnims = { kiki: "npc-kiki", moca: "npc-moca" };
  const roomTriggers = mapData.layers?.find((l) => l.name === "dialogue_triggers");
  if (roomTriggers) {
    for (const obj of roomTriggers.objects) {
      const anim = roomNpcAnims[obj.name];
      if (!anim) continue;
      const x = (obj.x + (obj.width || 0) / 2) * scaleFactor;
      const y = (obj.y + (obj.height || 0) / 2) * scaleFactor;
      k.add([
        k.sprite("spritesheet", { anim }),
        k.pos(x, y),
        k.anchor("center"),
        k.scale(scaleFactor),
        k.z(0),
      ]);
    }
  }

  setCamScale(k);
  k.onResize(() => setCamScale(k));
  addMovementAndCamera(player, {
    onStep: () => k.play("floor_walk", { volume: 0.7 }),
  });
});

k.go("island");
