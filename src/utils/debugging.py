import socket

import debugpy  # pip install debugpy
import torch.distributed as dist


def remote_breakpoint(host: str = "0.0.0.0", port: int = 5678, rank: int = 0):
    """
    This function helps to debug programs running in the remote computing node.

    In VSCode, you should add the configuration to the `.vscode/launch.json`, sth. like this 👇
    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Remote Attach",
                "type": "python",
                "request": "attach",
                "connect": {
                    "host": "<hostname>",
                    "port": 5678
                },
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "."
                    }
                ],
                "justMyCode": false
            }
        ]
    }
    ```

    Then, you could insert one line of code to the debugging position:
    ```python
    from smoe.utils.debugging import remote_breakpoint; remote_breakpoint()
    ```

    After the program starts and encounters the breakpoint, you could remote attach the debugger.

    Args:
        host: remote debug server's host (0.0.0.0 by default)
        port: remote debug server's port
        rank: which rank would you want to break at when debugging during distributed training
    """

    def _dp():
        print(
            f"Waiting for debugger to attach on {host}:{port}, server: {socket.gethostname()}..."
        )
        debugpy.listen((host, port))
        debugpy.wait_for_client()
        breakpoint()

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == rank:
            _dp()
        dist.barrier()
    else:
        _dp()
