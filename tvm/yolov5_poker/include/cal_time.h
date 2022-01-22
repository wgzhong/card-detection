#include <iostream>
#include <time.h>
#include <sys/time.h>    
#include <unistd.h> 
long getTimeUsec()
{
    struct timeval t;
    gettimeofday(&t, 0);
    return (long)((long)t.tv_sec * 1000 * 1000 + t.tv_usec);
}

