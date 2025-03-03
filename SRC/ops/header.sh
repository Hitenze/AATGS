echo "#ifndef AATGS_OPS_HEADER_H" > _ops.h
echo "#define AATGS_OPS_HEADER_H" >> _ops.h
echo '#include "_utils.h"' >> _ops.h
sed -n '12,$p' vecops.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _ops.h
sed -n '13,$p' matops.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _ops.h
echo "#endif" >> _ops.h
mv _ops.h ../../INC/_ops.h