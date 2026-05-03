import argparse
import importlib
import json
import multiprocessing as mp
import os
import platform
import time


def _square_worker(value):
    time.sleep(0.1)
    return {
        "input": value,
        "output": value * value,
        "pid": os.getpid(),
        "start_method": mp.get_start_method(),
    }


def _tensorflow_worker(value):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return {
        "input": value,
        "pid": os.getpid(),
        "tensorflow_version": tf.__version__,
        "gpu_count": len(gpus),
    }


def _module_import_worker(payload):
    value, module_name = payload
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return {
            "input": value,
            "pid": os.getpid(),
            "module": module_name,
            "module_version": version,
            "import_ok": True,
        }
    except Exception as exc:
        return {
            "input": value,
            "pid": os.getpid(),
            "module": module_name,
            "import_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_pool_test(start_method, worker, values, pool_size):
    ctx = mp.get_context(start_method)
    started = time.time()
    with ctx.Pool(pool_size) as pool:
        results = pool.map(worker, values)
    elapsed = time.time() - started
    return {
        "start_method": start_method,
        "pool_size": pool_size,
        "elapsed_seconds": round(elapsed, 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Small multiprocessing smoke test.")
    parser.add_argument(
        "--start-method",
        choices=mp.get_all_start_methods(),
        default="spawn",
        help="Multiprocessing start method to test.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=2,
        help="Number of worker processes to launch.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=4,
        help="Number of small tasks to submit to the pool.",
    )
    parser.add_argument(
        "--tensorflow",
        action="store_true",
        help="Import TensorFlow inside each worker to mimic the MASCOTS path more closely.",
    )
    parser.add_argument(
        "--module",
        help="Import this module inside each worker and report success or failure.",
    )
    parser.add_argument(
        "--import-parent",
        help="Import this module in the parent process before starting the pool.",
    )
    args = parser.parse_args()

    worker = _tensorflow_worker if args.tensorflow else _square_worker
    values = list(range(args.tasks))
    parent_import = None

    if args.import_parent:
        try:
            module = importlib.import_module(args.import_parent)
            parent_import = {
                "module": args.import_parent,
                "import_ok": True,
                "module_version": getattr(module, "__version__", None),
            }
        except Exception as exc:
            parent_import = {
                "module": args.import_parent,
                "import_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if args.module:
        worker = _module_import_worker
        values = [(value, args.module) for value in values]

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "available_start_methods": mp.get_all_start_methods(),
                "selected_start_method": args.start_method,
                "pool_size": args.pool_size,
                "tasks": args.tasks,
                "tensorflow_worker": args.tensorflow,
                "worker_module": args.module,
                "parent_import": parent_import,
            },
            indent=2,
        )
    )

    results = run_pool_test(args.start_method, worker, values, args.pool_size)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
