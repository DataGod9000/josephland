export function displayDialogue(text, onDisplayEnd) {
  const dialogueUI = document.getElementById("textbox-container");
  const dialogue = document.getElementById("dialogue");
  const closeBtn = document.getElementById("close");

  const pages = Array.isArray(text) ? text : [text];
  let pageIndex = 0;
  let intervalRef = null;

  function runTypewriter(pageText) {
    let index = 0;
    let currentText = "";
    if (intervalRef) clearInterval(intervalRef);
    intervalRef = setInterval(() => {
      if (index < pageText.length) {
        currentText += pageText[index];
        dialogue.innerHTML = currentText;
        index++;
        return;
      }
      clearInterval(intervalRef);
      intervalRef = null;
    }, 1);
  }

  function onCloseBtnClick() {
    pageIndex++;
    if (pageIndex < pages.length) {
      dialogue.innerHTML = "";
      closeBtn.textContent = pageIndex === pages.length - 1 ? "Close" : "Next";
      runTypewriter(pages[pageIndex]);
    } else {
      onDisplayEnd();
      dialogueUI.style.display = "none";
      dialogue.innerHTML = "";
      if (intervalRef) clearInterval(intervalRef);
      closeBtn.removeEventListener("click", onCloseBtnClick);
      closeBtn.textContent = "Close";
      window.removeEventListener("keydown", onKeyHandler);
    }
  }

  function onKeyHandler(e) {
    if (e.code === "Enter" || e.code === "Space") {
      closeBtn.click();
    }
  }

  dialogueUI.style.display = "block";
  closeBtn.textContent = pages.length > 1 ? "Next" : "Close";
  closeBtn.addEventListener("click", onCloseBtnClick);
  window.addEventListener("keydown", onKeyHandler);
  runTypewriter(pages[0]);
}

export function setCamScale(k) {
  const resizeFactor = k.width() / k.height();
  if (resizeFactor < 1) {
    k.camScale(k.vec2(1));
  } else {
    k.camScale(k.vec2(1.5));
  }
}
