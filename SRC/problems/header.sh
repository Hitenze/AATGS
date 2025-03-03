echo "#ifndef AATGS_PROBLEM_HEADER_H" > _problem.h
echo "#define AATGS_PROBLEM_HEADER_H" >> _problem.h
echo '#include "_utils.h"' >> _problem.h
echo '#include "_ops.h"' >> _problem.h
sed -n '12,$p' problem.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _problem.h
sed -n '13,$p' bratu.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _problem.h
sed -n '11,$p' lennardjones.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _problem.h
sed -n '14,$p' hequation.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _problem.h
echo "#endif" >> _problem.h
mv _problem.h ../../INC/_problem.h