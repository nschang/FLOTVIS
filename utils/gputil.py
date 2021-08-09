# Import packages
import os,sys,humanize,psutil,GPUtil
import time

# Define function
def mem_report():
  print 'CPU RAM Free: ' + humanize.naturalsize( psutil.virtual_memory().available ) , '               |', 'CPU_Utilization ' ,  psutil.cpu_percent(),'%'
  #print('CPU      ... RAM Free: {:}MB | Utilization {:}'.format(psutil.virtual_memory().available, psutil.cpu_percent()))
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | GPU_Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    
# Execute function for single report
#mem_report()

# recursive report
stored_exception=None

while True:
    try:
        mem_report()
        # do something time-cosnuming
        time.sleep(3)
        if stored_exception:
            break
    except KeyboardInterrupt:
        stored_exception=sys.exc_info()

if stored_exception:
    raise stored_exception[0], stored_exception[1], stored_exception[2]

sys.exit()
