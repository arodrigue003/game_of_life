
PLATFORM=1
export PLATFORM
KERNEL=compute_pixel_opt
export KERNEL
TILEX=32
export TILEX

PARAM="-n -s 1024 -i 10000 -v 7 -a" # parametres commun à toutes les executions


EXE="./prog $* $PARAM"
OUTPUT="$(echo $EXE | tr -d ' ')"
echo "$PARAM : " >> $OUTPUT;
for i in {1..20}
do
$EXE 2>> $OUTPUT;
done
