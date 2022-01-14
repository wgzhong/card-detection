#include <stdio.h>      /*标准输入输出定义*/    
#include <stdlib.h>     /*标准函数库定义*/    
#include <unistd.h>     /*Unix 标准函数定义*/    
#include <sys/types.h>     
#include <sys/stat.h>       
#include <fcntl.h>      /*文件控制定义*/    
#include <termios.h>    /*PPSIX 终端控制定义*/    
#include <errno.h>      /*错误号定义*/    
#include <string.h>    
#include <wiringSerial.h>
#include <tvm/runtime/logging.h>

#define FALSE  -1    
#define TRUE   0

class uart{
public:
    uart(){};
    bool uartInit(char *port, int bound){//"/dev/ttyAMA0"
        if ((fd = serialOpen(port, bound)) < 0)
        {
            fprintf(stderr, "Unable to open serial device: %s\n", strerror(errno));
            return FALSE;
        }
        // serialPuts(fd, "uart send test, just by launcher\n");
        return TRUE;
    }
    int uartRecv(char *recv_data){
        int len = serialDataAvail(fd);
        for (int i=0;i<len;i++){
            recv_data[i] = serialGetchar(fd);
        }
        return len;
    }
    void uartSend(const char *send_data, int len){
        for (int i=0;i<len;i++){
            putchar(send_data[i]);	
            serialPutchar(fd,send_data[i]);
        }
    }

private:
    int fd;
};

