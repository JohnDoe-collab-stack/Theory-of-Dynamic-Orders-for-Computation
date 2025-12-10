import Lake
open Lake DSL

package «LogicDissoc» where
  -- add package configuration options here

lean_lib «LogicDissoc» where
  srcDir := "LogicDissoc"
  roots := #[`LogicDissoc, `Sphere, `FYI, `Boole.AbstractKernel, `Boole.OmegaInstances]



require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
