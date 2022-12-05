from ctypes import (CDLL, CFUNCTYPE, POINTER, c_char_p, c_int, c_size_t,
                    c_void_p)

LIB = "kaldi_asr_client.so"

FUNCS = {
    "client_alloc": CFUNCTYPE(c_void_p),
    "client_infer_begin": CFUNCTYPE(c_int, c_void_p, c_size_t),
    "client_infer_feed": CFUNCTYPE(c_int, c_void_p, c_char_p, c_size_t),
    "client_infer_perform": CFUNCTYPE(POINTER(c_char_p), c_void_p),
    "client_infer_end": CFUNCTYPE(c_int, c_void_p),
    "client_destroy": CFUNCTYPE(None, c_void_p),
}


class Client:
    def __init__(self):
        c_lib = CDLL(LIB)

        for name, prototype in FUNCS.items():
            setattr(self, name, prototype(name, c_lib))

        self.client = c_lib.client_alloc()

    def infer(self, wavs: list[bytes]):
        # Preallocate vector space
        self.client_infer_begin(len(wavs))

        # Feed one by one to avoid dealing Python C API to extract data from
        # our object
        for wav in wavs:
            self.client_infer_feed(self.client, wav, len(wav))

        inferred = list(
            map(
                lambda cstr: cstr.decode(),
                self.client_infer_perform(self.client)[: len(wavs)],
            )
        )

        self.client_infer_end(self.client)

        return inferred

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client_destroy(self.client)
