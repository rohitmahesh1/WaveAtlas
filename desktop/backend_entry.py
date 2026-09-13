"""PyInstaller bootstrap: keep app imports after Windows spawn diversion."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from app.desktop.entry import main
    raise SystemExit(main())
