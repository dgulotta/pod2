use std::{
    array,
    marker::PhantomData,
    ops::{Index, Mul, Sub},
};

use plonky2::{
    field::{
        extension::{Extendable, FieldExtension, OEF},
        types::Field,
    },
    gates::gate::Gate,
    hash::hash_types::RichField,
    iop::{
        ext_target::ExtensionTarget,
        generator::{SimpleGenerator, WitnessGeneratorRef},
        target::Target,
        wire::Wire,
        witness::{Witness, WitnessWrite},
    },
    plonk::{circuit_builder::CircuitBuilder, circuit_data::CircuitConfig},
    util::serialization::{Read, Write},
};

use crate::backends::plonky2::primitives::ec::field::{CircuitBuilderNNF, OEFTarget};

#[derive(Debug)]
struct NNFMulGenerator<const DEG: usize, NNF: OEF<DEG>> {
    row: usize,
    slot: usize,
    _phantom_data: PhantomData<fn(NNF) -> NNF>,
}

impl<const DEG: usize, NNF: OEF<DEG>> NNFMulGenerator<DEG, NNF> {
    fn new(row: usize, slot: usize) -> Self {
        Self {
            row,
            slot,
            _phantom_data: PhantomData,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct TensorProduct<const F1_DEG: usize, F1, F2>
where
    F1: OEF<F1_DEG>,
    F2: Field + From<F1::BaseField>,
{
    pub components: [F2; F1_DEG],
    _phantom_data: PhantomData<F1>,
}

impl<const F1_DEG: usize, F1, F2> TensorProduct<F1_DEG, F1, F2>
where
    F1: OEF<F1_DEG>,
    F2: Field + From<F1::BaseField>,
{
    fn new(components: [F2; F1_DEG]) -> Self {
        Self {
            components,
            _phantom_data: PhantomData,
        }
    }
}

impl<const F1_DEG: usize, F1, F2> Mul<Self> for TensorProduct<F1_DEG, F1, F2>
where
    F1: OEF<F1_DEG>,
    F2: Field + From<F1::BaseField>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut components = array::from_fn(|_| F2::ZERO);
        for i in 0..F1_DEG {
            for j in 0..F1_DEG {
                let prod = self.components[i] * rhs.components[j];
                if i + j < F1_DEG {
                    components[i + j] += prod;
                } else {
                    components[i + j - F1_DEG] += prod * (F1::W).into()
                }
            }
        }
        Self::new(components)
    }
}

impl<const F1_DEG: usize, F1, F2> Sub<Self> for TensorProduct<F1_DEG, F1, F2>
where
    F1: OEF<F1_DEG>,
    F2: Field + From<F1::BaseField>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(array::from_fn(|i| self.components[i] - rhs.components[i]))
    }
}

impl<const NNF_DEG: usize, NNF, const D: usize, F> SimpleGenerator<F, D>
    for NNFMulGenerator<NNF_DEG, NNF>
where
    F: RichField + Extendable<D>,
    NNF: OEF<NNF_DEG> + FieldExtension<NNF_DEG, BaseField = F>,
{
    fn id(&self) -> String {
        "NNFMulGenerator".to_string()
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.slot)
    }

    fn deserialize(
        src: &mut plonky2::util::serialization::Buffer,
        _common_data: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self>
    where
        Self: Sized,
    {
        let row = src.read_usize()?;
        let slot = src.read_usize()?;
        Ok(Self::new(row, slot))
    }

    fn dependencies(&self) -> Vec<plonky2::iop::target::Target> {
        (0..(2 * NNF_DEG))
            .map(|i| {
                Target::Wire(Wire {
                    row: self.row,
                    column: self.slot * (3 * NNF_DEG) + i,
                })
            })
            .collect()
    }

    fn run_once(
        &self,
        witness: &plonky2::iop::witness::PartitionWitness<F>,
        out_buffer: &mut plonky2::iop::generator::GeneratedValues<F>,
    ) -> anyhow::Result<()> {
        let deps = <NNFMulGenerator<NNF_DEG, NNF> as SimpleGenerator<F, D>>::dependencies(self);
        let x = NNF::from_basefield_array(array::from_fn(|i| witness.get_target(deps[i])));
        let y =
            NNF::from_basefield_array(array::from_fn(|i| witness.get_target(deps[i + NNF_DEG])));
        let ans = (x * y).to_basefield_array();
        for (i, a) in ans.iter().enumerate() {
            out_buffer.set_target(
                Target::Wire(Wire {
                    row: self.row,
                    column: (self.slot * 3 + 2) * NNF_DEG + i,
                }),
                *a,
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NNFMulGate<const DEG: usize, NNF: OEF<DEG>> {
    max_ops: usize,
    _phantom_data: PhantomData<fn(NNF) -> NNF>,
}

impl<const DEG: usize, NNF: OEF<DEG>> NNFMulGate<DEG, NNF> {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            max_ops: config.num_routed_wires / (3 * DEG),
            _phantom_data: PhantomData,
        }
    }

    fn eval<const D: usize, I, F>(&self, wires: &I) -> Vec<I::Output>
    where
        F: Extendable<D>,
        I: Index<usize, Output = F::Extension> + ?Sized,
        NNF: FieldExtension<DEG, BaseField = F>,
    {
        let mut constraints = Vec::with_capacity(self.max_ops * DEG);
        for i in 0..self.max_ops {
            let xstart = 3 * DEG * i;
            let ystart = xstart + DEG;
            let zstart = ystart + DEG;
            let x: TensorProduct<DEG, NNF, <F as Extendable<D>>::Extension> =
                TensorProduct::new(array::from_fn(|i| wires[xstart + i]));
            let y = TensorProduct::new(array::from_fn(|i| wires[ystart + i]));
            let z = TensorProduct::new(array::from_fn(|i| wires[zstart + i]));
            let diff = x * y - z;
            for d in diff.components {
                constraints.push(d);
            }
        }
        constraints
    }
}

impl<NNF, const D: usize, const NNF_DEG: usize, F> Gate<F, D> for NNFMulGate<NNF_DEG, NNF>
where
    NNF: OEF<NNF_DEG> + FieldExtension<NNF_DEG, BaseField = F>,
    F: RichField + Extendable<D> + Extendable<1, Extension = F>,
    // this trait bound should automatically be satisfied but the compiler complains if it's removed
    CircuitBuilder<F, D>: CircuitBuilderNNF<F, D, NNF, OEFTarget<NNF_DEG, NNF>>,
{
    fn id(&self) -> String {
        "NNFMulGate".to_string()
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<()> {
        dst.write_usize(self.max_ops)
    }

    fn deserialize(
        src: &mut plonky2::util::serialization::Buffer,
        _common_data: &plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
    ) -> plonky2::util::serialization::IoResult<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            max_ops: src.read_usize()?,
            _phantom_data: PhantomData,
        })
    }

    fn degree(&self) -> usize {
        2
    }

    fn generators(
        &self,
        row: usize,
        _local_constants: &[F],
    ) -> Vec<plonky2::iop::generator::WitnessGeneratorRef<F, D>> {
        (0..self.max_ops)
            .map(|slot| {
                WitnessGeneratorRef::new(NNFMulGenerator::<NNF_DEG, NNF>::new(row, slot).adapter())
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.max_ops * 3 * NNF_DEG
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn num_constraints(&self) -> usize {
        self.max_ops * NNF_DEG
    }

    fn eval_unfiltered(
        &self,
        vars: plonky2::plonk::vars::EvaluationVars<F, D>,
    ) -> Vec<<F as Extendable<D>>::Extension> {
        self.eval::<D, _, _>(vars.local_wires)
    }

    fn eval_unfiltered_base_one(
        &self,
        vars_base: plonky2::plonk::vars::EvaluationVarsBase<F>,
        mut yield_constr: plonky2::gates::util::StridedConstraintConsumer<F>,
    ) {
        let constraints = self.eval::<1, _, _>(&vars_base.local_wires);
        yield_constr.many(constraints)
    }

    #[allow(clippy::needless_range_loop)]
    fn eval_unfiltered_circuit(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: plonky2::plonk::vars::EvaluationTargets<D>,
    ) -> Vec<plonky2::iop::ext_target::ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.max_ops * NNF_DEG);
        for i in 0..self.max_ops {
            let xstart = 3 * NNF_DEG * i;
            let ystart = xstart + NNF_DEG;
            let zstart = ystart + NNF_DEG;
            let target_array = |start: usize| {
                array::from_fn(|j| {
                    OEFTarget::<NNF_DEG, NNF>::new(array::from_fn(|k| {
                        vars.local_wires[start + k].to_target_array()[j]
                    }))
                })
            };
            let x: [_; D] = target_array(xstart);
            let y: [_; D] = target_array(ystart);
            let mut z: [_; D] = target_array(zstart);
            for j in 0..D {
                for k in 0..D {
                    let tmp = builder.nnf_mul(&x[j], &y[k]);
                    let mut idx = j + k;
                    if idx < D {
                        z[idx] = builder.nnf_sub(&z[idx], &tmp);
                    } else {
                        idx -= D;
                        let w = builder.constant(NNF::W);
                        let tmp2 = builder.nnf_mul_scalar(w, &tmp);
                        z[idx] = builder.nnf_sub(&z[idx], &tmp2);
                    }
                }
            }
            for j in 0..NNF_DEG {
                constraints.push(ExtensionTarget(array::from_fn(|k| z[k].components[j])));
            }
        }
        constraints
    }
}
