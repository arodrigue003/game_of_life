
PLATFORM=1
export PLATFORM
KERNEL=compute_pixel_opt
export KERNEL
TILEX=32
export TILEX

PARAM="-n -s 4096 -i 10000 -v 9" # parametres commun à toutes les executions


EXE="./prog $* $PARAM"
OUTPUT="$(echo $EXE | tr -d ' ')"
echo "$TILEX $KERNEL $PARAM : " >> $OUTPUT;
for i in {1..5}
do
$EXE 2>> $OUTPUT;
done
