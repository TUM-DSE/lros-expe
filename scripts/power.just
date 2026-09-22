import? "../justfile"


# OrangePi Notes
# powerlogger.py for the power measurements and sensors for the temperature
# Stop it with `sudo kill -15 [pid of powerlogger.py]` (SIGTERM)

measure_orangepi:
	#!/usr/bin/env bash
	sudo PYTHONPATH=$PYTHONPATH ./powerlogger.py


# Jetson: temperatures in /sys/devices/virtual/thermal/thermal_zone* (0 CPU, 1 GPU,
# 8 the hottest; GPU and CV zones read -EAGAIN while their rail is off). Power from
# the INA3221 monitors, /sys/bus/i2c/drivers/ina3221/1-004[01]/hwmon/hwmon7/:
# 0x40 ch1 GPU+SoC, ch2 CPU, ch3 system (5V); 0x41 ch2 DRAM.

measure_jetson period="1":
	#!/usr/bin/env bash
	base_0x40="/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon7/"
	base_0x41="/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon7/"
	base_temp="/sys/devices/virtual/thermal/"
	beginning=$(date "+%s.%N")
	
	make_measurement(){
		t=$(bc <<< "$(date "+%s.%N") - $beginning")
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
		echo ${t::-6},$sys_v,$sys_i,$sys_p,$sys_temp0,$sys_temp1,$sys_temp2,$cpu_v,$cpu_i,$cpu_p,$cpu_temp,$gpu_v,$gpu_i,$gpu_p,$gpu_temp,$mem_v,$mem_i,$mem_p
	}

	echo "timestamp,system_voltage,system_intensity,system_power,system_temp0,system_temp1,system_temp2,cpu_voltage,cpu_intensity,cpu_power,cpu_temp,gpu_voltage,gpu_intensity,gpu_power,gpu_temp,memory_voltage,memory_intensity,memory_power"
	while true
	do
		make_measurement
		sleep {{period}}
	done
