cmake_minimum_required(VERSION 3.14)
project(test)


#指定编译工具链
set(CMAKE_C_COMPILER "arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-none-eabi-g++")
set(CMAKE_ASM_COMPILER "arm-none-eabi-gcc -x assembler-with-cpp")
set(CMAKE_AR "arm-none-eabi-ar")
set(CMAKE_OBJCOPY "arm-none-eabi-objcopy")
set(CMAKE_OBJDUMP "arm-none-eabi-objdump")
set(CMAKE_SIZE "arm-none-eabi-size")
set(HEX "${CMAKE_OBJCOPY} -O ihex")
set(BIN "${CMAKE_OBJCOPY} -O binary -S")
#编译相关选项
set(DEBUG "1")#是否开启debug
set(OPT "-Og") #编译优化等级
set(BUILD_DIR "build") #编译目录
set(ASM_SOURCES "startup_stm32f407xx.s") #.s启动文件

set(CPU "-mcpu=cortes-m4")#cpu
set(FPU "-mfpu=fpv4-sp-d16")#fpu 浮点运算单元
set(FLOAT_ABI "-mfloat-abi=hard")
set(MCU "${CPU} -mthumb ${FPU} ${FLOAT_ABI}") #MCU


#设置宏定义
add_definitions(-DUSE_HAL_DRIVER -DSTM32F407xx)
#设置头文件包含路径
include_directories(
    Core/Inc 
    Drivers/STM32F4xx_HAL_Driver/Inc 
    Drivers/STM32F4xx_HAL_Driver/Inc/Legacy 
    Drivers/CMSIS/Device/ST/STM32F4xx/Include 
    Drivers/CMSIS/Include 
)

#startup文件是.s文件，这里需要说明
ENABLED_LANGUAGES(ASM)


set(SRC_STARTUP "${ASM_SOURCES}")
aux_source_directory(Core/Src)
aux_source_directory(Drivers/STM32F4xx_HAL_Driver)

#连接生成
set()