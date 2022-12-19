import socket

HOST = "localhost"
PORT = 5555


def restart_servers(n_servers):
    if type(n_servers) != int:
        raise TypeError("n_servers must be an integer")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))
        client.send(bytes(str(n_servers), "utf-8") + b"\n")

        ret = int(client.recv(1))

        if ret != 0:
            raise Exception(
                "Daemon failed to restart servers, check server logs"
            )
