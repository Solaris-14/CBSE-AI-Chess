// Drag & drop implementation, removed animations. Shows full move list (white & black).
let selectedSquare = null;
let boardState = null;      // fen string
let prevFenMap = null;
const files = "abcdefgh";
let dragging = null;        // { origin, imgEl, offsetX, offsetY }

// Create the 8x8 board grid
function createBoard() {
    const board = document.getElementById("board");
    board.innerHTML = "";

    for (let rank = 7; rank >= 0; rank--) {
        for (let file = 0; file < 8; file++) {

            const square = document.createElement("div");
            square.classList.add("square");

            const isLight = (rank + file) % 2 === 0;
            square.classList.add(isLight ? "light" : "dark");

            const squareName = files[file] + (rank + 1);
            square.dataset.square = squareName;

            // rank label at left side of each row (show on first file)
            if (file === 0) {
                const rlabel = document.createElement("div");
                rlabel.className = "coord rank";
                rlabel.textContent = (rank + 1).toString();
                square.appendChild(rlabel);
            }

            // file label on bottom row
            if (rank === 0) {
                const flabel = document.createElement("div");
                flabel.className = "coord file";
                flabel.textContent = files[file];
                square.appendChild(flabel);
            }

            // click fallback for non-drag usage
            square.addEventListener("click", () => onSquareClick(squareName));

            board.appendChild(square);
        }
    }

    // pointer events for drag-n-drop
    board.addEventListener("pointerdown", onPointerDown);
}

// helper: prompt promotion chooser — returns one of 'q','r','b','n'
function promptPromotion(color = "w") {
    return new Promise((resolve) => {
        // create overlay
        const overlay = document.createElement("div");
        overlay.className = "promo-overlay";

        const box = document.createElement("div");
        box.className = "promo-box";

        const opts = [
            {k: "q", label: "Queen"},
            {k: "r", label: "Rook"},
            {k: "b", label: "Bishop"},
            {k: "n", label: "Knight"}
        ];

        opts.forEach(o => {
            const btn = document.createElement("button");
            btn.className = "promo-option";
            // try to show piece image if available
            const img = document.createElement("img");
            img.src = `assets/pieces/${color}${o.k}.png`;
            img.alt = o.k;
            img.width = 48;
            img.height = 48;
            btn.appendChild(img);

            const lbl = document.createElement("div");
            lbl.className = "promo-label";
            lbl.textContent = o.label;
            btn.appendChild(lbl);

            btn.addEventListener("click", () => {
                document.body.removeChild(overlay);
                resolve(o.k);
            });
            box.appendChild(btn);
        });

        // clicking outside defaults to queen
        overlay.addEventListener("click", (ev) => {
            if (ev.target === overlay) {
                document.body.removeChild(overlay);
                resolve("q");
            }
        });

        overlay.appendChild(box);
        document.body.appendChild(overlay);
    });
}

// small helper to detect promotion necessity
function needsPromotion(origAlt, destSquare) {
    if (!origAlt) return false;
    const piece = origAlt; // char like 'P' or 'p'
    if (piece.toLowerCase() !== "p") return false;
    const destRank = parseInt(destSquare[1], 10);
    if (piece === piece.toUpperCase() && destRank === 8) return true; // white pawn to rank 8
    if (piece === piece.toLowerCase() && destRank === 1) return true; // black pawn to rank 1
    return false;
}

function parseFen(fen) {
    const map = {};
    const rows = fen.split(" ")[0].split("/");
    for (let r = 0; r < 8; r++) {
        let row = rows[r];
        let file = 0;
        for (let ch of row) {
            if (!isNaN(parseInt(ch))) {
                file += parseInt(ch);
            } else {
                const square = files[file] + (8 - r);
                map[square] = ch;
                file++;
            }
        }
    }
    return map;
}

function renderPieces(fen) {
    boardState = fen;

    // Clear existing pieces
    document.querySelectorAll(".square").forEach(sq => sq.innerHTML = "");

    let rows = fen.split(" ")[0].split("/");

    for (let rank = 7; rank >= 0; rank--) {
        let row = rows[7 - rank];
        let file = 0;

        for (let char of row) {
            if (Number.isInteger(parseInt(char))) {
                file += parseInt(char);
                continue;
            }

            let pieceColor = (char === char.toUpperCase()) ? "w" : "b";
            let pieceType = char.toLowerCase();

            let squareName = files[file] + (rank + 1);
            file++;

            let img = document.createElement("img");
            img.classList.add("piece");
            img.src = `assets/pieces/${pieceColor}${pieceType}.png`;
            img.alt = char;
            img.draggable = false; // use pointer drag

            document.querySelector(`[data-square="${squareName}"]`).appendChild(img);
        }
    }

    // store mapping for next diff
    prevFenMap = parseFen(fen);
}


// Handle pointer down for drag start
function onPointerDown(e) {
    // only start drag if a piece was pressed
    const piece = e.target.closest(".piece");
    if (!piece) return;

    e.preventDefault();

    const sqEl = piece.parentElement;
    const origin = sqEl.dataset.square;
    const rect = piece.getBoundingClientRect();

    // create drag image
    const clone = piece.cloneNode(true);
    clone.classList.add("drag-piece");
    clone.style.width = `${rect.width}px`;
    clone.style.height = `${rect.height}px`;
    document.body.appendChild(clone);

    // hide original piece while dragging
    piece.style.opacity = "0";

    dragging = {
        origin,
        origEl: piece,
        imgEl: clone,
        offsetX: e.clientX - rect.left,
        offsetY: e.clientY - rect.top
    };

    moveDragImage(e.clientX, e.clientY);

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
}

function moveDragImage(clientX, clientY) {
    if (!dragging) return;
    const x = clientX - dragging.offsetX;
    const y = clientY - dragging.offsetY;
    dragging.imgEl.style.left = `${x}px`;
    dragging.imgEl.style.top = `${y}px`;
}

function onPointerMove(e) {
    if (!dragging) return;
    e.preventDefault();
    moveDragImage(e.clientX, e.clientY);
}

async function onPointerUp(e) {
    if (!dragging) return;

    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);

    // find drop target square
    let dropSquare = null;
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const sq = el ? el.closest(".square") : null;
    if (sq) dropSquare = sq.dataset.square;

    const origin = dragging.origin;
    const origEl = dragging.origEl;
    const imgEl = dragging.imgEl;

    // remove drag image
    imgEl.remove();
    dragging = null;

    if (!dropSquare) {
        // cancelled -> restore original
        origEl.style.opacity = "1";
        return;
    }

    // same square -> restore and do nothing
    if (dropSquare === origin) {
        origEl.style.opacity = "1";
        return;
    }

    // promotion check
    if (needsPromotion(origEl.alt, dropSquare)) {
        const color = (origEl.alt === origEl.alt.toUpperCase()) ? "w" : "b";
        const promo = await promptPromotion(color);
        const move = origin + dropSquare + promo;
        document.getElementById("status").textContent = `Trying ${move}`;
        try {
            const result = await eel.make_human_move(move)();
            if (result === "invalid") {
                document.getElementById("status").textContent = "Invalid move!";
                origEl.style.opacity = "1";
                return;
            }
            if (result.startsWith("game_over")) {
                document.getElementById("status").textContent = result;
                await refreshBoard();
                return;
            }
            await refreshBoard();
            await aiMove();
        } catch (err) {
            console.error("make_human_move error", err);
            origEl.style.opacity = "1";
        }
        return;
    }

    // attempt move via backend (UCI origin+dest)
    const move = origin + dropSquare;
    document.getElementById("status").textContent = `Trying ${move}`;
    try {
        const result = await eel.make_human_move(move)();
        if (result === "invalid") {
            document.getElementById("status").textContent = "Invalid move!";
            // restore original piece
            origEl.style.opacity = "1";
            return;
        }

        if (result.startsWith("game_over")) {
            document.getElementById("status").textContent = result;
            await refreshBoard();
            return;
        }

        // successful human move — refresh and let AI play
        await refreshBoard();
        await aiMove();
    } catch (err) {
        console.error("make_human_move error", err);
        origEl.style.opacity = "1";
    }
}

// click-click fallback, handle promotions
async function onSquareClick(square) {
    if (selectedSquare === null) {
        selectedSquare = square;
        document.getElementById("status").textContent = `Selected ${square}`;
        return;
    }

    // If second click → attempt move
    let origin = selectedSquare;
    let dest = square;
    selectedSquare = null;

    // attempt promotion if needed
    const originEl = document.querySelector(`[data-square="${origin}"] .piece`);
    if (originEl && needsPromotion(originEl.alt, dest)) {
        const color = (originEl.alt === originEl.alt.toUpperCase()) ? "w" : "b";
        const promo = await promptPromotion(color);
        const move = origin + dest + promo;
        await makeHumanMove(move);
        return;
    }

    const move = origin + dest;
    await makeHumanMove(move);
}


async function makeHumanMove(move) {
    // kept for click-click fallback
    let result = await eel.make_human_move(move)();

    if (result === "invalid") {
        document.getElementById("status").textContent = "Invalid move!";
        return;
    }

    if (result.startsWith("game_over")) {
        document.getElementById("status").textContent = result;
        await refreshBoard();
        return;
    }

    // Human move was OK → refresh board & call AI
    await refreshBoard();
    await aiMove();
}


async function aiMove() {
    let aiMoveRes = await eel.ai_move()();

    if (aiMoveRes.startsWith("game_over")) {
        document.getElementById("status").textContent = aiMoveRes;
        await refreshBoard();
        return;
    }

    document.getElementById("status").textContent = `AI played ${aiMoveRes}`;
    await refreshBoard();
}


// add move list UI update (two-column style: white / black)
async function updateMoveList() {
    try {
        const moves = await eel.get_moves()(); // array of SAN strings
        const list = document.getElementById("moves-list");
        list.innerHTML = "";

        for (let i = 0; i < moves.length; i += 2) {
            const moveNumber = Math.floor(i / 2) + 1;
            const whiteMove = moves[i] || "";
            const blackMove = moves[i + 1] || "";
            const li = document.createElement("li");
            li.className = "move-row";
            li.innerHTML = `<span class="move-num">${moveNumber}.</span>
                            <span class="white-move">${whiteMove}</span>
                            <span class="black-move">${blackMove}</span>`;
            list.appendChild(li);
        }
        // scroll to bottom
        list.scrollTop = list.scrollHeight;
    } catch (e) {
        console.error("Failed to update moves:", e);
    }

    // update performance panel
    try {
        const perf = await eel.get_performance()();
        document.getElementById("stat-wins").textContent = perf.counts.wins;
        document.getElementById("stat-losses").textContent = perf.counts.losses;
        document.getElementById("stat-draws").textContent = perf.counts.draws;
        document.getElementById("stat-total").textContent = perf.counts.total;

        const recent = perf.recent_games || [];
        const recentEl = document.getElementById("perf-recent");
        recentEl.innerHTML = "<div style='margin-top:8px;font-size:12px;color:#cbd5e1;'>Recent</div>";
        recent.slice().reverse().forEach(g => {
            const d = new Date(g.ts * 1000);
            const node = document.createElement("div");
            node.style.fontSize = "12px";
            node.style.opacity = "0.9";
            node.textContent = `${d.toISOString().slice(0,19).replace("T"," ")} — ${g.result} (${g.moves} moves)`;
            recentEl.appendChild(node);
        });
    } catch (e) {
        console.warn("perf fetch failed", e);
    }
}


async function refreshBoard() {
    let fen = await eel.get_board_fen()();
    renderPieces(fen);
    updateMoveList();
    const board = document.getElementById("board");
    board.classList.remove("board-fade");
    void board.offsetWidth;
    board.classList.add("board-fade");
}


// Parallax: update CSS vars on mouse move for subtle depth
window.addEventListener("mousemove", (e) => {
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight / 2;
    const px = (e.clientX - cx) / cx; // -1..1
    const py = (e.clientY - cy) / cy;
    document.documentElement.style.setProperty('--px', (px * 20).toFixed(2));
    document.documentElement.style.setProperty('--py', (py * 20).toFixed(2));
});


// Initial load
createBoard();
refreshBoard();
document.getElementById("status").textContent = "Your move";

// wire reset button
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("resetBtn");
    if (btn) {
        btn.addEventListener("click", async () => {
            await eel.reset_game()();
            await refreshBoard();
            document.getElementById("status").textContent = "Game reset";
        });
    }
});
