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
#include "util.h"

#define FALSE  -1    
#define TRUE   0

class uart{
public:
    uart(int dsize){
        data_size = dsize;
        recv_data = new char[data_size];
        memset(recv_data, 0, data_size); 
    };
    bool uartInit(char *port, int bound){//"/dev/ttyAMA0"
        if ((fd = serialOpen(port, bound)) < 0)
        {
            LOG(ERROR)<<"Unable to open serial device: "<<strerror(errno);
            return FALSE;
        }
        serialPuts(fd, "uart send test, just by launcher\n");
        return TRUE;
    }
    bool uartRecv(){
        int len = read(fd, recv_data, data_size);
        if(len != data_size){
            char error_msg[data_size];
            memset(error_msg, 0xff, data_size);
            write(fd, error_msg, data_size);
            LOG(ERROR)<<"recv data not equal size="<<data_size;
            return FALSE;
        }
        return TRUE;
    }

    void uartSend(std::vector<ground_truth> send_buf, std::vector<std::string> labels, int topk=0){
        char send_data[data_size];
        char data[data_size-4-4];
        memset(send_data, 0x0, data_size);
        memcpy(send_data, head, 4);
        memcpy(send_data+data_size-4, tail, 4);
        for(int i=0; i<topk+1; i++){
            memcpy(data, &send_buf[i], sizeof(send_buf[i]));
            memcpy(send_data+4, data, data_size-4-4);
            write(fd, send_data, data_size);
            serialPrintf(fd, labels[send_buf[i].label_idx]);
        }
    }
    bool isCorrect(){
        char h[4],t[4],data[data_size-4-4];
        memcpy(h, recv_data, 4);
        memcpy(data, recv_data+4, data_size-4-4);
        memcpy(t, recv_data+data_size-4, 4);
        if((strcmp(head, h)!=0) || (strcmp(tail, t)!=0)){
            return FALSE;
        }
        return TRUE;
    }

    void clearData(){
        memset(recv_data, 0, data_size); 
    }

private:
    int fd;
    int data_size;
    char *recv_data;
    char head[4] = {0x21, 0x09, 0x01, 0x01};
    char tail[4] = {0x21, 0x09, 0x30, 0x01};
};

