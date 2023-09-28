for x in 4 5 6 7 8
do
  echo $x

  ./linux-exe/generator_swap $x 2
  ./linux-exe/generator_swap $x 3
  ./linux-exe/generator_swap $x 4
  ./linux-exe/generator_swap $x $x swap-n

  ./linux-exe/generator_insert $x 3
  ./linux-exe/generator_insert $x 4
  ./linux-exe/generator_insert $x $x insert-n

  ./linux-exe/generator_reverse $x 4
  ./linux-exe/generator_reverse $x $x reverse-n

  ./linux-exe/generator_block $x 4
  ./linux-exe/generator_block $x $x block-n
done
