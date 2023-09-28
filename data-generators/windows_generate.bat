for /l %%x in (4, 1, 8) do (
  echo %%x

  .\windows-exe\generator_swap.exe %%x 2
  .\windows-exe\generator_swap.exe %%x 3
  .\windows-exe\generator_swap.exe %%x 4
  .\windows-exe\generator_swap %%x %%x swap-n

  .\windows-exe\generator_insert.exe %%x 3
  .\windows-exe\generator_insert.exe %%x 4
  .\windows-exe\generator_insert.exe %%x %%x insert-n

  .\windows-exe\generator_reverse.exe %%x 4
  .\windows-exe\generator_reverse.exe %%x %%x reverse-n

  .\windows-exe\generator_block.exe %%x 4
  .\windows-exe\generator_block.exe %%x %%x block-n
)
