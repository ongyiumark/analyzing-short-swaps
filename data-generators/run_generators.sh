for x in 4 5 6 7
do
  echo $x

  ./swap-gen $x 2
  ./swap-gen $x 3
  ./swap-gen $x $x nswap

  ./insert-gen $x 2
  ./insert-gen $x 3
  ./insert-gen $x $x ninsert

  ./reverse-gen $x 2
  ./reverse-gen $x 3
  ./reverse-gen $x $x nreverse
done
