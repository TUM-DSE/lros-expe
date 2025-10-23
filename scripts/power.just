import? "../justfile"

# OrangePi Notes
# powerlogger.py for the power measurements and sensors for the temperature
#

measure_orangepi:
	#!/usr/bin/env bash
	sudo env PATH=$PATH ./powerlogger.py

# Jetson Notes
# `sensors` for the power and temperature measurements
# Important values: VDD_GPU_SOC and VDD_CPU_CV for the voltage and curr for the intensity
# Nvidia documentation (for temperature): https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#thermal-management 
# Thermal zones in /sys/devices/virtual/thermal/thermal_zone*
# 0 is cpu-thermal, 1 is gpu, 2-3-4 are for the computer vision (CV) cores, 5-6-7 for different points of the SoC (???) and 8 is a virtual sensor corresponding to the highest temp between all other sensors.
# GPU and CV cores will only be available if the power rail is turned on. Otherwise querying sysfs will result in a -EAGAIN error.
# Nvidia does not share the location of the sensors for the general SoC. Could be possible to reverse-engineer the location with stress testing. For now, maybe enough to simply report all and possibly average these values
#
# Power is reported by the INA3221 monitors at IC2 addresses 0x40 and 0x41 (https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling)
# In the sysfs at /sys/bus/i2c/drivers/ina3221/1-004[0/1]/hwmon/hwmon7/
# At 0x40: channel 1 is the GPU and the SoC, channel 2 is the CPU, channel 3 is the whole system (at 5V)
# At 0x41: channel 1 and 3 are unused, channel 2 is the DRAM

measure_jetson:
	#!/usr/bin/env bash
	base_0x40="/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon7/"
	base_0x41="/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon7/"
	base_temp="/sys/devices/virtual/thermal/"
	
	make_measurement(){
		sys_v=$(bc <<< "scale=3; $(cat $base_0x40/in3_input)/1000")
		sys_i=$(bc <<< "scale=3; $(cat $base_0x40/curr3_input)/1000")
		sys_p=$(bc <<< "scale=3; $sys_v*$sys_i")
		sys_temp0=$(bc <<< "scale=2; $(cat $base_temp/thermal_zone5/temp)/1000")
		sys_temp1=$(bc <<< "scale=2; $(cat $base_temp/thermal_zone6/temp)/1000")
		sys_temp2=$(bc <<< "scale=2; $(cat $base_temp/thermal_zone7/temp)/1000")
		cpu_v=$(bc <<< "scale=3; $(cat $base_0x40/in1_input)/1000")
		cpu_i=$(bc <<< "scale=3; $(cat $base_0x40/curr1_input)/1000")
		cpu_p=$(bc <<< "scale=3; $cpu_v*$cpu_i")
		cpu_temp=$(bc <<< "scale=2; $(cat $base_temp/thermal_zone0/temp)/1000")
		gpu_v=$(bc <<< "scale=3; $(cat $base_0x40/in2_input)/1000")
		gpu_i=$(bc <<< "scale=3; $(cat $base_0x40/curr2_input)/1000")
		gpu_p=$(bc <<< "scale=3; $gpu_v*$gpu_i")
		gpu_temp_raw=$(cat $base_temp/thermal_zone1/temp 2> /dev/null)
		if [ "$?" = "0" ]; then
			gpu_temp=$(bc <<< "scale=2; $gpu_temp_raw/1000")
		else
			gpu_temp=""
		fi
		mem_v=$(bc <<< "scale=3; $(cat $base_0x41/in2_input)/1000")
		mem_i=$(bc <<< "scale=3; $(cat $base_0x41/curr2_input)/1000")
		mem_p=$(bc <<< "scale=3; $mem_v*$mem_i")
		echo $sys_v,$sys_i,$sys_p,$sys_temp0,$sys_temp1,$sys_temp2,$cpu_v,$cpu_i,$cpu_p,$cpu_temp,$gpu_v,$gpu_i,$gpu_p,$gpu_temp,$mem_v,$mem_i,$mem_p
	}

	echo "system_voltage,system_intensity,system_power,system_temp0,system_temp1,system_temp2,cpu_voltage,cpu_intensity,cpu_power,cpu_temp,gpu_voltage,gpu_intensity,gpu_power,gpu_temp,memory_voltage,memory_intensity,memory_power"
	for i in `seq 1 30`
	do
		make_measurement
		sleep 1
	done
