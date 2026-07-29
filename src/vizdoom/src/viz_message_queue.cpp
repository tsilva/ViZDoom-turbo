/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include "d_main.h"
#include "viz_message_queue.h"
#include "viz_input.h"
#include "viz_game.h"
#include "viz_main.h"

#include <errno.h>
#include <stddef.h>
#if defined(__linux__)
#include <linux/futex.h> //VIZDOOM_CODE
#include <sys/syscall.h> //VIZDOOM_CODE
#include <unistd.h> //VIZDOOM_CODE
#endif

EXTERN_CVAR (Int, viz_debug)
EXTERN_CVAR (Bool, viz_async)

bip::message_queue *vizMQController = nullptr;
bip::message_queue *vizMQDoom = nullptr;
char *vizMQControllerName;
char *vizMQDoomName;
unsigned int vizPendingTics = 0;
unsigned int vizBatchTicsMade = 0;
bool vizBatchUpdate = false;
bool vizTurboReset = false;
bool vizTurboResetStarted = false;
unsigned int vizTurboResetInitialTic = 0;
unsigned int vizTurboResetTargetTic = 0;
bool vizFastIPC = false; //VIZDOOM_CODE
uint32_t vizFastCommandSequence = 0; //VIZDOOM_CODE

#if defined(__linux__)
//VIZDOOM_CODE
static void VIZ_FastWait(uint32_t *address, uint32_t observed) {
    while (__atomic_load_n(address, __ATOMIC_ACQUIRE) == observed) {
        if (syscall(SYS_futex, address, FUTEX_WAIT, observed, NULL, NULL, 0) == -1 &&
            errno != EAGAIN && errno != EINTR) {
            VIZ_Error(VIZ_FUNC, "Failed to wait for Turbo IPC.");
        }
    }
}

//VIZDOOM_CODE
static void VIZ_FastWake(uint32_t *address) {
    if (syscall(SYS_futex, address, FUTEX_WAKE, 1, NULL, NULL, 0) == -1) {
        VIZ_Error(VIZ_FUNC, "Failed to wake Turbo IPC.");
    }
}
#endif

void VIZ_MQInit(const char * id){

    Printf("VIZ_MQInit: Init message queues.\n");

	vizMQControllerName = new char[strlen(VIZ_MQ_NAME_CTR_BASE) + strlen(id) + 1];
	strcpy(vizMQControllerName, VIZ_MQ_NAME_CTR_BASE);
	strcat(vizMQControllerName, id);

	vizMQDoomName = new char[strlen(VIZ_MQ_NAME_DOOM_BASE) + strlen(id) +1];
	strcpy(vizMQDoomName, VIZ_MQ_NAME_DOOM_BASE);
	strcat(vizMQDoomName, id);

    try{
        vizMQController = new bip::message_queue(bip::open_only, vizMQControllerName);//, VIZ_MQ_MAX_MSG_NUM, VIZ_MQ_MAX_MSG_SIZE);
        vizMQDoom = new bip::message_queue(bip::open_only, vizMQDoomName);//, VIZ_MQ_MAX_MSG_NUM, VIZ_MQ_MAX_MSG_SIZE);
    }
    catch(...){ // bip::interprocess_exception
        VIZ_Error(VIZ_FUNC, "Failed to open message queues.");
    }
}


void VIZ_MQSend(uint8_t code, const char * command, uint32_t value){
#if defined(__linux__)
    //VIZDOOM_CODE
    if(vizFastIPC) {
        vizGameStateSM->FAST_RESPONSE_CODE = code;
        vizGameStateSM->FAST_RESPONSE_VALUE = value;
        memset(vizGameStateSM->FAST_RESPONSE, 0, sizeof(vizGameStateSM->FAST_RESPONSE));
        if(command) {
            strncpy(
                vizGameStateSM->FAST_RESPONSE,
                command,
                sizeof(vizGameStateSM->FAST_RESPONSE) - 1);
        }
        __atomic_store_n(
            &vizGameStateSM->FAST_DONE_SEQUENCE,
            vizFastCommandSequence,
            __ATOMIC_RELEASE);
        VIZ_FastWake(&vizGameStateSM->FAST_DONE_SEQUENCE);
        return;
    }
#endif
    VIZMessage msg = {};
    msg.code = code;
    msg.value = value;
    if(command) strncpy(msg.command, command, VIZ_MQ_MAX_CMD_LEN);

    if(vizMQController) {
        vizMQController->send(
            &msg,
            command ? sizeof(VIZMessage) : offsetof(VIZMessage, command),
            0);
    }

    VIZ_DebugMsg(4, VIZ_FUNC, "Sent msg: %d.", code);
}

void VIZ_MQReceive(void *msg) {
#if defined(__linux__)
    //VIZDOOM_CODE
    if(vizFastIPC) {
        uint32_t sequence;
        while((sequence = __atomic_load_n(
                   &vizGameStateSM->FAST_COMMAND_SEQUENCE,
                   __ATOMIC_ACQUIRE)) == vizFastCommandSequence) {
            VIZ_FastWait(&vizGameStateSM->FAST_COMMAND_SEQUENCE, sequence);
        }

        VIZMessage *message = static_cast<VIZMessage *>(msg);
        memset(message, 0, sizeof(*message));
        message->code = vizGameStateSM->FAST_COMMAND_CODE;
        message->value = vizGameStateSM->FAST_COMMAND_VALUE;
        strncpy(
            message->command,
            vizGameStateSM->FAST_COMMAND,
            sizeof(message->command) - 1);
        vizFastCommandSequence = sequence;
        __atomic_store_n(
            &vizGameStateSM->FAST_RECEIVED_SEQUENCE,
            sequence,
            __ATOMIC_RELEASE);
        VIZ_FastWake(&vizGameStateSM->FAST_RECEIVED_SEQUENCE);
        return;
    }
#endif
    if(vizMQDoom) {
        size_t size;
        unsigned int priority;

        try{            
            vizMQDoom->receive(msg, sizeof(VIZMessage), size, priority);
        }
        catch(...){ // bip::interprocess_exception
            VIZ_Error(VIZ_FUNC, "Failed to receive message.");
        }

        VIZ_DebugMsg(4, VIZ_FUNC, "Received msg: %d.", static_cast<VIZMessage *>(msg)->code);
    }
}

bool VIZ_MQTryReceive(void *msg){
#if defined(__linux__)
    //VIZDOOM_CODE
    if(vizFastIPC) {
        const uint32_t sequence = __atomic_load_n(
            &vizGameStateSM->FAST_COMMAND_SEQUENCE,
            __ATOMIC_ACQUIRE);
        if(sequence == vizFastCommandSequence) return false;

        VIZMessage *message = static_cast<VIZMessage *>(msg);
        memset(message, 0, sizeof(*message));
        message->code = vizGameStateSM->FAST_COMMAND_CODE;
        message->value = vizGameStateSM->FAST_COMMAND_VALUE;
        strncpy(
            message->command,
            vizGameStateSM->FAST_COMMAND,
            sizeof(message->command) - 1);
        vizFastCommandSequence = sequence;
        __atomic_store_n(
            &vizGameStateSM->FAST_RECEIVED_SEQUENCE,
            sequence,
            __ATOMIC_RELEASE);
        VIZ_FastWake(&vizGameStateSM->FAST_RECEIVED_SEQUENCE);
        return true;
    }
#endif
    size_t size;
    unsigned int priority;

    return vizMQDoom->try_receive(msg, sizeof(VIZMessage), size, priority);
}

void VIZ_MQTic(){

    VIZMessage msg;

    do {
        if(!*viz_async) VIZ_MQReceive(&msg);
        else if(!VIZ_MQTryReceive(&msg)) break;

        switch(msg.code){
            case VIZ_MSG_CODE_TIC :
                vizNextTic = true;
                break;

            case VIZ_MSG_CODE_UPDATE:
                VIZ_Update();
                VIZ_GameStateTic();
                VIZ_MQSend(VIZ_MSG_CODE_DOOM_DONE);
                break;

            case VIZ_MSG_CODE_TIC_AND_UPDATE:
                vizUpdate = true;
                vizNextTic = true;
                break;

            case VIZ_MSG_CODE_TICS:
            case VIZ_MSG_CODE_TICS_AND_UPDATE:
                vizPendingTics = msg.value;
                if (vizPendingTics == 0) vizPendingTics = 1;
                vizBatchTicsMade = 0;
                vizBatchUpdate = msg.code == VIZ_MSG_CODE_TICS_AND_UPDATE;
                vizUpdate = vizBatchUpdate && vizPendingTics == 1;
                vizNextTic = true;
                break;

            case VIZ_MSG_CODE_TURBO_RESET:
                vizTurboReset = true;
                vizTurboResetStarted = false;
                vizTurboResetInitialTic = level.maptime;
                vizTurboResetTargetTic = msg.value;
                vizNextTic = true;
                break;

            //VIZDOOM_CODE
            case VIZ_MSG_CODE_TURBO_FAST_IPC:
#if defined(__linux__)
                VIZ_MQSend(VIZ_MSG_CODE_DOOM_DONE);
                vizGameStateSM->FAST_COMMAND_SEQUENCE = 0;
                vizGameStateSM->FAST_RECEIVED_SEQUENCE = 0;
                vizGameStateSM->FAST_DONE_SEQUENCE = 0;
                vizFastCommandSequence = 0;
                vizFastIPC = true;
#else
                VIZ_MQSend(VIZ_MSG_CODE_DOOM_DONE);
#endif
                break;

            case VIZ_MSG_CODE_COMMAND:
                if(msg.command[0] != '\0') VIZ_Command(strdup(msg.command));
                VIZ_CVARsUpdate();
                break;

            case VIZ_MSG_CODE_CLOSE:
            case VIZ_MSG_CODE_ERROR:
                D_ClearAll();
                exit(0);

            default : break;
        }
    } while(!vizNextTic);
}

void VIZ_MQClose(){
    //bip::message_queue::remove(vizMQControllerName);
    //bip::message_queue::remove(vizMQDoomName);
    delete vizMQController;
    delete vizMQDoom;
	delete[] vizMQControllerName;
	delete[] vizMQDoomName;
}
