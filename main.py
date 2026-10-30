"""Primary web entry point. The preserved desktop launcher is desktop.py."""
import os


def main():
    os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=int(os.environ.get("TA_EFL_PORT", "8780")), workers=1, access_log=False)


if __name__ == "__main__":
    main()
