_plot script csv out category:
	#!/usr/bin/env bash
	csv_path="$(cd "{{invocation_directory()}}" && realpath "{{csv}}")"
	out="{{out}}"
	[ -z "$out" ] && out="${csv_path%.csv}_{{category}}.pdf"
	cd {{source_directory()}}/plots
	python3 {{script}} "$csv_path" "$out"

plot_parallel_sweep csv out="": (_plot "parallel_sweep.py" csv out "throughput")

plot_latency_sweep csv out="": (_plot "latency_sweep.py" csv out "latency")

plot_kv_memory csv out="": (_plot "kv_memory.py" csv out "kv_memory")

# Plot context-switch latency as a grouped bar chart (complete / preempt / resume).
plot_ctx_switch csv out="": (_plot "ctx_switch.py" csv out "ctx_switch")
