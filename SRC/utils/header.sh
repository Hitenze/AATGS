echo "#ifndef AATGS_UTILS_HEADER_H" > _utils.h
echo "#define AATGS_UTILS_HEADER_H" >> _utils.h
sed -n '8,$p' utils.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D' >> _utils.h
sed -n '10,$p' memory.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _utils.h
sed -n '10,$p' protos.h | sed -e :a -e '$d;N;2,2ba' -e 'P;D'>> _utils.h
echo "#endif" >> _utils.h
mv _utils.h ../../INC/_utils.h