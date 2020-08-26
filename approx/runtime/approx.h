extern "C" {
void __approx_exec_call(void (*accurate)(void *), void (*perforate)(void *),
                        void *arg, unsigned char cond, void *perfoArgs, void *deps,
                        int num_deps);
}
