import Lake
open Lake DSL

package «LogicDissoc» where
  -- add package configuration options here

lean_lib «LogicDissoc» where
  srcDir := "LogicDissoc"
  roots := #[`LogicDissoc]

@[default_target]
lean_exe «logicdissoc» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
