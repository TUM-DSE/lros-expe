#pragma once
#include <unistd.h>
#include <sys/syscall.h>

static inline long hvc_trace(unsigned long id)
{
    return syscall(468, id);
}