use std::marker::PhantomData;

use num::BigUint;
use plonky2::{
    field::{
        extension::{Extendable, FieldExtension, OEF},
        types::Field,
    },
    hash::hash_types::RichField,
    iop::{
        generator::{GeneratedValues, SimpleGenerator},
        target::{BoolTarget, Target},
        witness::{PartitionWitness, Witness, WitnessWrite},
    },
    plonk::{circuit_builder::CircuitBuilder, circuit_data::CommonCircuitData},
    util::serialization::{Buffer, IoError, Read, Write},
};

use super::gates::field::NNFMulGate;
use crate::{backends::plonky2::basetypes::D, middleware::F};

/// Trait for incorporating non-native field (NNF) arithmetic into a
/// circuit.
pub trait CircuitBuilderNNF<
    F: RichField + Extendable<D>,
    const D: usize,
    NNF: Field,
    NNFTarget: Clone,
>
{
    // Target 'adder'
    fn add_virtual_nnf_target(&mut self) -> NNFTarget;
    // Constant introducers.
    fn nnf_constant(&mut self, x: &NNF) -> NNFTarget;
    fn nnf_zero(&mut self) -> NNFTarget {
        self.nnf_constant(&NNF::ZERO)
    }
    fn nnf_one(&mut self) -> NNFTarget {
        self.nnf_constant(&NNF::ONE)
    }

    // Field ops
    fn nnf_add(&mut self, x: &NNFTarget, y: &NNFTarget) -> NNFTarget;
    fn nnf_sub(&mut self, x: &NNFTarget, y: &NNFTarget) -> NNFTarget;
    fn nnf_mul(&mut self, x: &NNFTarget, y: &NNFTarget) -> NNFTarget;
    fn nnf_div(&mut self, x: &NNFTarget, y: &NNFTarget) -> NNFTarget {
        let y_inv = self.nnf_inverse(y);
        self.nnf_mul(x, &y_inv)
    }
    fn nnf_inverse(&mut self, x: &NNFTarget) -> NNFTarget {
        let one = self.nnf_one();
        self.nnf_div(&one, x)
    }
    fn nnf_mul_generator(&mut self, x: &NNFTarget) -> NNFTarget;
    fn nnf_mul_scalar(&mut self, x: Target, y: &NNFTarget) -> NNFTarget;
    fn nnf_add_scalar_times_generator_power(
        &mut self,
        x: Target,
        gen_power: usize,
        y: &NNFTarget,
    ) -> NNFTarget;
    fn nnf_if(&mut self, b: BoolTarget, x_true: &NNFTarget, x_false: &NNFTarget) -> NNFTarget;
    fn nnf_exp_biguint(&mut self, base: &NNFTarget, exponent: &BigUint) -> NNFTarget;

    // Equality check and connection
    fn nnf_eq(&mut self, x: &NNFTarget, y: &NNFTarget) -> BoolTarget;
    fn nnf_connect(&mut self, x: &NNFTarget, y: &NNFTarget);
}

/// Target type modelled on OEF.
#[derive(Debug, Clone)]
pub struct OEFTarget<const DEG: usize, NNF: OEF<DEG>> {
    pub components: [Target; DEG],
    _phantom_data: PhantomData<NNF>,
}

impl<const DEG: usize, NNF: OEF<DEG>> OEFTarget<DEG, NNF> {
    pub fn new(components: [Target; DEG]) -> Self {
        Self {
            components,
            _phantom_data: PhantomData,
        }
    }
}

impl<const DEG: usize, NNF: OEF<DEG>> Default for OEFTarget<DEG, NNF> {
    fn default() -> Self {
        Self::new([Target::default(); DEG])
    }
}

/// Quotient generator for OEF targets. Allows us to automagically
/// generate quotients as witnesses.
#[derive(Debug, Default)]
struct QuotientGeneratorOEF<const DEG: usize, NNF: OEF<DEG>> {
    numerator: OEFTarget<DEG, NNF>,
    denominator: OEFTarget<DEG, NNF>,
    quotient: OEFTarget<DEG, NNF>,
}

impl<
        const DEG: usize,
        NNF: OEF<DEG> + FieldExtension<DEG, BaseField = F>,
        F: RichField + Extendable<D>,
        const D: usize,
    > SimpleGenerator<F, D> for QuotientGeneratorOEF<DEG, NNF>
{
    fn id(&self) -> String {
        "QuotientGeneratorOEF".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.numerator
            .components
            .iter()
            .chain(self.denominator.components.iter())
            .cloned()
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<(), anyhow::Error> {
        // Dereference numerator & denominator targets to vectors
        // and construct field elements.
        let num_components = self
            .numerator
            .components
            .iter()
            .map(|t| witness.get_target(*t))
            .collect::<Vec<_>>();
        let den_components = self
            .denominator
            .components
            .iter()
            .map(|t| witness.get_target(*t))
            .collect::<Vec<_>>();

        let num = NNF::from_basefield_array(std::array::from_fn(|i| num_components[i]));
        let den = NNF::from_basefield_array(std::array::from_fn(|i| den_components[i]));

        let quotient = num / den;

        out_buffer.set_target_arr(&self.quotient.components, &quotient.to_basefield_array())
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> Result<(), IoError> {
        dst.write_target_array(&self.numerator.components)?;
        dst.write_target_array(&self.denominator.components)?;
        dst.write_target_array(&self.quotient.components)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D>,
    ) -> Result<Self, IoError> {
        let numerator = OEFTarget::new(src.read_target_array()?);
        let denominator = OEFTarget::new(src.read_target_array()?);
        let quotient = OEFTarget::new(src.read_target_array()?);
        Ok(Self {
            numerator,
            denominator,
            quotient,
        })
    }
}

impl<const DEG: usize, NNF: OEF<DEG> + FieldExtension<DEG, BaseField = F>>
    CircuitBuilderNNF<F, D, NNF, OEFTarget<DEG, NNF>> for CircuitBuilder<F, D>
{
    fn add_virtual_nnf_target(&mut self) -> OEFTarget<DEG, NNF> {
        OEFTarget::new(self.add_virtual_target_arr())
    }
    fn nnf_constant(&mut self, x: &NNF) -> OEFTarget<DEG, NNF> {
        let targets = x
            .to_basefield_array()
            .iter()
            .map(|c| self.constant(*c))
            .collect::<Vec<_>>();
        OEFTarget::new(std::array::from_fn(|i| targets[i]))
    }
    fn nnf_add(&mut self, x: &OEFTarget<DEG, NNF>, y: &OEFTarget<DEG, NNF>) -> OEFTarget<DEG, NNF> {
        let sum_targets = std::iter::zip(&x.components, &y.components)
            .map(|(a, b)| self.add(*a, *b))
            .collect::<Vec<_>>();
        OEFTarget::new(std::array::from_fn(|i| sum_targets[i]))
    }
    fn nnf_sub(&mut self, x: &OEFTarget<DEG, NNF>, y: &OEFTarget<DEG, NNF>) -> OEFTarget<DEG, NNF> {
        let sub_targets = std::iter::zip(&x.components, &y.components)
            .map(|(a, b)| self.sub(*a, *b))
            .collect::<Vec<_>>();
        OEFTarget::new(std::array::from_fn(|i| sub_targets[i]))
    }
    fn nnf_mul(&mut self, x: &OEFTarget<DEG, NNF>, y: &OEFTarget<DEG, NNF>) -> OEFTarget<DEG, NNF> {
        let gate = NNFMulGate::<DEG, NNF>::new_from_config(&self.config);
        let (row, slot) = self.find_slot(gate, &[], &[]);
        let col_start = 3 * DEG * slot;
        for i in 0..DEG {
            self.connect(Target::wire(row, col_start + i), x.components[i]);
            self.connect(Target::wire(row, col_start + DEG + i), y.components[i]);
        }
        OEFTarget::new(std::array::from_fn(|i| {
            Target::wire(row, col_start + 2 * DEG + i)
        }))
    }
    fn nnf_div(&mut self, x: &OEFTarget<DEG, NNF>, y: &OEFTarget<DEG, NNF>) -> OEFTarget<DEG, NNF> {
        // Determine quotient witness from numerator & denominator
        // witnesses.
        let quotient = self.add_virtual_nnf_target();
        self.add_simple_generator(QuotientGeneratorOEF {
            numerator: x.clone(),
            denominator: y.clone(),
            quotient: quotient.clone(),
        });

        // Add constraints.
        let quotient_times_denominator = self.nnf_mul(&quotient, y);
        self.nnf_connect(x, &quotient_times_denominator);

        quotient
    }
    fn nnf_mul_generator(&mut self, x: &OEFTarget<DEG, NNF>) -> OEFTarget<DEG, NNF> {
        OEFTarget::new(std::array::from_fn(|i| {
            if i == 0 {
                self.mul_const(NNF::W, x.components[DEG - 1])
            } else {
                x.components[i - 1]
            }
        }))
    }
    fn nnf_mul_scalar(&mut self, x: Target, y: &OEFTarget<DEG, NNF>) -> OEFTarget<DEG, NNF> {
        OEFTarget::new(std::array::from_fn(|i| self.mul(x, y.components[i])))
    }
    fn nnf_add_scalar_times_generator_power(
        &mut self,
        x: Target,
        gen_power: usize,
        y: &OEFTarget<DEG, NNF>,
    ) -> OEFTarget<DEG, NNF> {
        OEFTarget::new(std::array::from_fn(|i| {
            if i == gen_power {
                self.add(x, y.components[i])
            } else {
                y.components[i]
            }
        }))
    }
    fn nnf_if(
        &mut self,
        b: BoolTarget,
        x_true: &OEFTarget<DEG, NNF>,
        x_false: &OEFTarget<DEG, NNF>,
    ) -> OEFTarget<DEG, NNF> {
        OEFTarget::new(std::array::from_fn(|i| {
            self._if(b, x_true.components[i], x_false.components[i])
        }))
    }
    fn nnf_exp_biguint(
        &mut self,
        base: &OEFTarget<DEG, NNF>,
        exponent: &BigUint,
    ) -> OEFTarget<DEG, NNF> {
        let mut ans = self.nnf_one();
        for i in (0..exponent.bits()).rev() {
            ans = self.nnf_mul(&ans, &ans);
            if exponent.bit(i) {
                ans = self.nnf_mul(&ans, base);
            }
        }
        ans
    }
    fn nnf_eq(&mut self, x: &OEFTarget<DEG, NNF>, y: &OEFTarget<DEG, NNF>) -> BoolTarget {
        let eq_checks = std::iter::zip(&x.components, &y.components)
            .map(|(a, b)| self.is_equal(*a, *b))
            .collect::<Vec<_>>();
        eq_checks
            .into_iter()
            .reduce(|check, c| self.and(check, c))
            .expect("Missing equality checks")
    }
    fn nnf_connect(&mut self, x: &OEFTarget<DEG, NNF>, y: &OEFTarget<DEG, NNF>) {
        std::iter::zip(&x.components, &y.components).for_each(|(a, b)| self.connect(*a, *b))
    }
}

pub(super) fn get_nnf_target<const DEG: usize, NNF: OEF<DEG>>(
    witness: &impl Witness<NNF::BaseField>,
    tgt: &OEFTarget<DEG, NNF>,
) -> NNF {
    let values = tgt.components.map(|x| witness.get_target(x));
    NNF::from_basefield_array(values)
}

#[cfg(test)]
mod test {
    use plonky2::{
        field::{
            extension::quintic::QuinticExtension,
            goldilocks_field::GoldilocksField,
            types::{Field, Sample},
        },
        iop::witness::{PartialWitness, WitnessWrite},
        plonk::{
            circuit_builder::CircuitBuilder, circuit_data::CircuitConfig,
            config::PoseidonGoldilocksConfig,
        },
    };

    use super::{CircuitBuilderNNF, OEFTarget};

    #[test]
    fn quintic_arithmetic_check() -> Result<(), anyhow::Error> {
        type QuinticGoldilocks = QuinticExtension<GoldilocksField>;

        // Circuit declaration
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<GoldilocksField, 2>::new(config);

        let zero = builder.nnf_zero();
        let one = builder.nnf_one();

        // Let c = a * b.
        let a_target: OEFTarget<5, QuinticGoldilocks> = builder.add_virtual_nnf_target();
        let b_target: OEFTarget<5, QuinticGoldilocks> = builder.add_virtual_nnf_target();
        let c_target = builder.nnf_mul(&a_target, &b_target);

        // Pick some values.
        let a_value = QuinticExtension(std::array::from_fn(|_| GoldilocksField::rand()));
        let b_value = {
            let rand_value = QuinticExtension(std::array::from_fn(|_| GoldilocksField::rand()));
            if rand_value == QuinticGoldilocks::ZERO {
                QuinticGoldilocks::ONE
            } else {
                rand_value
            }
        };
        let c_value = a_value * b_value;

        // How about d = a/b?
        let d_target = builder.nnf_div(&a_target, &b_target);
        let d_value = a_value / b_value;

        // Also e = a - b.
        let e_target = builder.nnf_sub(&a_target, &b_target);
        let e_value = a_value - b_value;

        // a +- 0 == a, a * 1 == a, etc.
        let a_plus_zero = builder.nnf_add(&a_target, &zero);
        let a_minus_zero = builder.nnf_sub(&a_target, &zero);
        let a_times_one = builder.nnf_mul(&a_target, &one);
        let a_div_one = builder.nnf_div(&a_target, &one);

        builder.nnf_connect(&a_target, &a_plus_zero);
        builder.nnf_connect(&a_target, &a_minus_zero);
        builder.nnf_connect(&a_target, &a_times_one);
        builder.nnf_connect(&a_target, &a_div_one);

        // a == a, a != a + 1
        let a_plus_one = builder.nnf_add(&a_target, &one);
        let a_eq_a = builder.nnf_eq(&a_target, &a_target);
        let a_eq_a_plus_one = builder.nnf_eq(&a_target, &a_plus_one);

        builder.assert_one(a_eq_a.target);
        builder.assert_zero(a_eq_a_plus_one.target);

        // b * (1/b) == 1
        let one_on_b = builder.nnf_inverse(&b_target);
        let b_times_one_on_b = builder.nnf_mul(&b_target, &one_on_b);

        builder.nnf_connect(&one, &b_times_one_on_b);

        // Prove
        let mut pw = PartialWitness::<GoldilocksField>::new();

        pw.set_target_arr(&a_target.components, &a_value.0)?;
        pw.set_target_arr(&b_target.components, &b_value.0)?;
        pw.set_target_arr(&c_target.components, &c_value.0)?;
        pw.set_target_arr(&d_target.components, &d_value.0)?;
        pw.set_target_arr(&e_target.components, &e_value.0)?;

        let data = builder.build::<PoseidonGoldilocksConfig>();
        let proof = data.prove(pw)?;
        data.verify(proof)?;

        Ok(())
    }
}
