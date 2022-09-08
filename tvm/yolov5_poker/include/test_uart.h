#include <stdio.h>      /*标准输入输出定义*/    
#include <stdlib.h>     /*标准函数库定义*/    
#include <unistd.h>     /*Unix 标准函数定义*/    
#include <sys/types.h>     
#include <sys/stat.h>       
#include <fcntl.h>      /*文件控制定义*/    
#include <termios.h>    /*PPSIX 终端控制定义*/    
#include <errno.h>      /*错误号定义*/    
#include <string.h>    

#if defined  __aarch64__
    #include <wiringSerial.h>
#endif
#include <tvm/runtime/logging.h>
#include "util.h"

#define FALSE  0    
#define TRUE   1

class uart{
public:
    uart(int rcv, int send){
        recv_size = rcv;
        send_size = send;
        recv_flag = FALSE;
        recv_data = new char[recv_size];
        memset(recv_data, 0, recv_size);    
    };
    bool uartInit(char *port, int bound){//"/dev/ttyAMA0"
        struct termios stNew;
        struct termios stOld;
        #if defined  __aarch64__
            fd = serialOpen(port,bound);     
        #elif defined __x86_64__
            fd = open(port, O_RDWR|O_NOCTTY|O_NONBLOCK|O_NDELAY);
        #endif
        if(-1 == fd)
        {
            LOG(ERROR)<<"Unable to open port serial device: "<<strerror(errno);
            return FALSE;
        }
        if( (fcntl(fd, F_SETFL, O_NONBLOCK)) < 0 ) 
        {
            LOG(ERROR)<<"Fcntl F_SETFL Error: "<<strerror(errno);
            return FALSE;
        }
        if(tcgetattr(fd, &stOld) != 0)
        {
            LOG(ERROR)<<"tcgetattr error: "<<strerror(errno);
            return FALSE;
        }

        stNew = stOld;
        cfmakeraw(&stNew);//将终端设置为原始模式，该模式下全部的输入数据以字节为单位被处理
        //set speed
        cfsetispeed(&stNew, bound);//115200
        cfsetospeed(&stNew, bound);
        //set databits  
        stNew.c_cflag |= (CLOCAL|CREAD);
        stNew.c_cflag &= ~CSIZE;
        stNew.c_cflag |= CS8;
        //set parity  
        stNew.c_cflag &= ~PARENB;  
        stNew.c_iflag &= ~INPCK;
        //set stopbits  
        stNew.c_cflag &= ~CSTOPB;
        stNew.c_cc[VTIME]=0;	//指定所要读取字符的最小数量
        stNew.c_cc[VMIN]=1;	    //指定读取第一个字符的等待时间，时间的单位为n*100ms
                    //假设设置VTIME=0，则无字符输入时read（）操作无限期的堵塞
        tcflush(fd,TCIFLUSH);	//清空终端未完毕的输入/输出请求及数据。
        if( tcsetattr(fd,TCSANOW,&stNew) != 0 )
        {
            perror("tcsetattr Error!\n");
            return FALSE;
        }
        write(fd, (char*)"uart init ok!!!\n", sizeof("uart init ok!!!\n"));
        return TRUE;
    }

    bool uartRecv(){
        int len = read(fd, recv_data, recv_size);
		if(len <= 0)
		{
			return FALSE;
		}
        if(len != recv_size){
            char error_msg[recv_size];
            memset(error_msg, 0xff, recv_size);
            write(fd, error_msg, recv_size);
            memset(recv_data, 0, recv_size); 
            LOG(ERROR)<<"recv data not equal size recv_size!=len"<<recv_size <<"!= "<<len;
            clearData();
            recv_flag = FALSE;
            tcflush(fd,TCIFLUSH);
            return FALSE;
        }
        recv_flag = TRUE;
        return TRUE;
    }

    void uartSend(std::vector<ground_truth> send_buf, std::vector<std::string> labels, int topk=0){
        char send_data[send_size];
        char data[send_size-4-4];
        memset(send_data, 0x0, send_size);
        memcpy(send_data, head, 4);
        memcpy(send_data+send_size-4, tail, 4);
        if(send_buf.size() == 0){
            ground_truth gt;
            send_buf.push_back(gt);
        }
        if(send_buf.size()>0){
            for(int i=0; i<topk+1; i++){
                memcpy(data, &send_buf[i], sizeof(send_buf[i]));
                memcpy(send_data+4, data, send_size-4-4);
                write(fd, send_data, send_size);
                std::string cls = labels[send_buf[i].label_idx]+"\n";
                write(fd, cls.c_str(), cls.size());
            }
        }
    }
    bool isCorrect(){
        if(recv_flag){
            for(int i=0;i<recv_size;i++){
                if(i<4 && recv_data[i] != head[i]){
                    clearData();
                    char error_msg[recv_size];
                    memset(error_msg, 0xee, recv_size);
                    write(fd, error_msg, recv_size);
                    return FALSE;
                } else if(i>3 && recv_data[i] != tail[i-4]){
                    char error_msg[recv_size];
                    memset(error_msg, 0xee, recv_size);
                    write(fd, error_msg, recv_size);
                    clearData();
                    return FALSE;
                }
            }
            clearData();
            return TRUE;
        }
        clearData();
        return FALSE;
    }

    void clearData(){
        memset(recv_data, 0, recv_size);
        tcflush(fd,TCIFLUSH);
    }

private:
    int fd;
    int recv_size;
    int send_size;
    char *recv_data;
    bool recv_flag;
    char head[4] = {0x21, 0x09, 0x01, 0x01};
    char tail[4] = {0x21, 0x09, 0x30, 0x01};
};

