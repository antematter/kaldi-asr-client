from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_void_p,
    c_bool,
)

from functools import partial

LIB = "./build/libkaldi-asr-parallel-client.so"

FUNCS = {
    "client_alloc": CFUNCTYPE(c_void_p),
    "client_set_config": CFUNCTYPE(
        c_int,
        c_void_p,
        c_float,
        POINTER(c_char_p),
        c_char_p,
        c_int,
        c_int,
        c_bool,
    ),
    "client_infer_begin": CFUNCTYPE(c_int, c_void_p, c_size_t),
    "client_infer_feed": CFUNCTYPE(c_int, c_void_p, c_char_p, c_size_t),
    "client_infer_perform": CFUNCTYPE(c_int, c_void_p),
    "client_infer_output": CFUNCTYPE(c_char_p, c_void_p),
    "client_last_error": CFUNCTYPE(c_char_p, c_void_p),
    "client_destroy": CFUNCTYPE(None, c_void_p),
}


class Client:
    def __init__(
        self,
        samp_freq,
        servers=["localhost:8001"],
        model_name="kaldi_online",
        ncontextes=10,
        chunk_length=8160,
        verbose=False,
    ):
        c_lib = CDLL(LIB)

        for name, prototype in FUNCS.items():
            func = prototype((name, c_lib))

            if func.restype == c_int:
                setattr(self, name, partial(self.wrap_exc, func))
            else:
                setattr(self, name, func)

        # NULL terminated array
        servers_cstr = (c_char_p * (len(servers) + 1))()

        for idx, server in enumerate(servers):
            servers_cstr[idx] = bytes(server, "utf-8")

        servers_cstr[len(servers)] = None

        self.client = self.client_alloc()
        self.client_set_config(
            self.client,
            samp_freq,
            servers_cstr,
            bytes(model_name, "utf-8"),
            ncontextes,
            chunk_length,
            verbose,
        )

    def wrap_exc(self, func, *args, **kwargs):
        if (result := func(*args, **kwargs)) != 0:
            assert result == -1
            raise Exception(self.client_last_error(self.client).decode())

        return result

    def infer(self, wavs: list[bytes]):
        # Preallocate vector space
        self.client_infer_begin(self.client, len(wavs))

        # Feed one by one to avoid dealing Python C API to extract data from
        # our object
        for wav in wavs:
            if type(wav) != bytes:
                raise TypeError("WAVs must be bytes objects!")

            self.client_infer_feed(self.client, wav, len(wav))

        self.client_infer_perform(self.client)

        inferred = []

        while (cstr := self.client_infer_output(self.client)) != None:
            inferred.append(cstr.decode())

        assert len(inferred) == len(wavs)

        return inferred

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client_destroy(self.client)
