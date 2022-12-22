import socket


def restart_servers(servers, host="localhost", port=5555):
    data = ",".join(
        map(lambda server: str(int(server.split(":")[-1])), servers)
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((host, port))
        client.send(bytes(data, "utf-8") + b"\n")

        ret = int(client.recv(1))

        if ret != 0:
            raise Exception(
                "Daemon failed to restart servers, check server logs"
            )
