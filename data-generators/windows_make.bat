Set dir="./windows-exe/"
if not exist %dir% mkdir %dir%
g++ converter_cli.cpp headers/converter.cpp -o %dir%/converter_cli -O3
g++ generator_swap.cpp headers/converter.cpp headers/generator.cpp -o %dir%/generator_swap -O3
g++ generator_insert.cpp headers/converter.cpp headers/generator.cpp -o %dir%/generator_insert -O3
g++ generator_reverse.cpp headers/converter.cpp headers/generator.cpp -o %dir%/generator_reverse -O3
g++ generator_block.cpp headers/converter.cpp headers/generator.cpp -o %dir%/generator_block -O3