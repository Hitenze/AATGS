echo "#ifndef AATGS_OPTIMIZER_HEADER_H" > _optimizer.h
echo "#define AATGS_OPTIMIZER_HEADER_H" >> _optimizer.h
echo '#include "_utils.h"' >> _optimizer.h
echo '#include "_ops.h"' >> _optimizer.h
echo '#include "_problem.h"' >> _optimizer.h
sed -n '13,$p' optimizer.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _optimizer.h
sed -n '13,$p' gd.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _optimizer.h
sed -n '13,$p' adam.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _optimizer.h
sed -n '13,$p' anderson.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _optimizer.h
sed -n '13,$p' nltgcr.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _optimizer.h
echo "#endif" >> _optimizer.h
mv _optimizer.h ../../INC/_optimizer.h