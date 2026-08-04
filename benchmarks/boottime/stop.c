#include <poll.h>
#include <unistd.h>

static int fd = -1;

/**
 * We dup stdin so we can still stop on shutdown when usually the file streams
 * are already closed by the libc.
 */
__attribute__((constructor))
static void startup() {
    fd = dup(0);
}

/**
 * Stop on shutdown so we can read final memory statistics.
 * Released by a byte on stdin or by stdin being closed (EOF).
 */
__attribute__((destructor))
static void shutdown() {
    struct pollfd p = { .fd = fd, .events = POLLIN };
    if (fd < 0)
        return;
    /* poll instead of read: qemu's console may have switched stdin to
     * non-blocking, which the dup'd fd shares. poll pauses regardless until
     * input is available or the other end is closed. */
    poll(&p, 1, -1);
    close(fd);
}