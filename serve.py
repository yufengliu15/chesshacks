
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
import os

from src.utils import chess_manager
from src import main

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.get("/api/bot")
@app.post("/api/bot")
async def bot_info():
    """Bot info endpoint - confirms the bot is connected and available."""
    return JSONResponse(content={
        "name": "Transformer Chess Bot",
        "version": "1.0",
        "model": "transformer",
        "ready": True
    })


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    
    # Show which checkpoint is currently loaded
    checkpoint_info = main._ENGINE._load_model.__self__ if hasattr(main._ENGINE, '_load_model') else None
    print(f"\n{'='*70}")
    print(f"🎮 Move Request Received")
    print(f"{'='*70}")
    print(f"📋 PGN: {pgn}")
    
    # Try to get checkpoint info from engine
    try:
        if hasattr(main._ENGINE, 'model') and main._ENGINE.model is not None:
            # Get checkpoint path from engine's config
            checkpoint_path = os.environ.get("CHESS_CHECKPOINT", "AUTO (latest in checkpoint dir)")
            print(f"🎯 Active Checkpoint: {checkpoint_path}")
            
            # If available, show step number
            if checkpoint_path != "AUTO (latest in checkpoint dir)":
                step_num = checkpoint_path.split("step_")[-1].split(".")[0] if "step_" in checkpoint_path else "unknown"
                print(f"📊 Checkpoint Step: {step_num}")
            
            # Show MCTS status
            if main._ENGINE.config.use_mcts:
                print(f"🌳 MCTS: Enabled ({main._ENGINE.config.mcts_simulations} sims)")
            else:
                print(f"⚡ MCTS: Disabled (greedy selection)")
    except:
        pass
    
    print(f"{'='*70}\n")

    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=500,
        )

    # Confirm type of move_probs
    if not isinstance(move_probs, dict):
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(content={m: None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
