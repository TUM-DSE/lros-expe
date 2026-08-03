#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>

static int fd;

/**
 * To be able to attach to qemu and trace it we need to get its PID and stop until we are attached.
 * We dup stdin so we can also stop on shutdown when usually the file streams are already closed by the libc.
 */
__attribute__((constructor))
static void startup() {
    char selfpath[128];
    memset(selfpath, 0, sizeof (selfpath));
    if (readlink("/proc/self/exe", selfpath, sizeof (selfpath) - 1)==-1){
        perror("Read path of executable.");
    }
    // Nix wraps qemu to set some env variables.
    // We want to pause once we reached the actual executable.
    // Its name is .qemu-system-aarch64-wrapped.
    if (basename(selfpath)[0] == '.') {
        printf("%ld\n", (long) getpid());
        fd = dup(0);
        getchar();
    }
}

/**
 * Stop on shutdown so we can read final memory statistics.
 */
__attribute__((destructor))
static void shutdown() {
    char c;
    /* Blocks until at least 1 byte is available or EOF
     * Ignore output. */
    (void) !read(fd, &c, 1);
    close(fd);
}
