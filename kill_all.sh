lsof /dev/nvidia0  | awk  '{print $2}' |xargs -I {} kill -9 {}
lsof /dev/nvidia1  | awk  '{print $2}' |xargs -I {} kill -9 {}
lsof /dev/nvidia2  | awk  '{print $2}' |xargs -I {} kill -9 {}
lsof /dev/nvidia3  | awk  '{print $2}' |xargs -I {} kill -9 {}
lsof /dev/nvidia4  | awk  '{print $2}' |xargs -I {} kill -9 {}
lsof /dev/nvidia5  | awk  '{print $2}' |xargs -I {} kill -9 {}
lsof /dev/nvidia6  | awk  '{print $2}' |xargs -I {} kill -9 {}
lsof /dev/nvidia7  | awk  '{print $2}' |xargs -I {} kill -9 {}
