import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the local Object Inpainting Dataset Studio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run("object_inpainting_studio.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
