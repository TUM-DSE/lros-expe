#include <stdio.h>
#include <stdlib.h>
#include "outb.h"

/* Access the benchmark port from the guest user space.
 * The host can record this events using a tracer such as bpftrace.
 */

#define EVENT_USER 240

int main(int argc, char *argv[]) {
    unsigned char value = EVENT_USER;
    if (argc >= 2) {
        value = (char)atoi(argv[1]);
    }

    if (hvc_trace(value)) return 1;
    return 0;
}
