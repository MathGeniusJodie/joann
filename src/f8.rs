#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct f8(u8);

impl BitAnd for f8 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        f8(self.0 & rhs.0)
    }
}

impl From<f8> for f32 {
    fn from(f: f8) -> f32 {
        let sign = f.0 & 0b1000_0000;
        let exponent = f.0 & 0b0111_1000;
        let mantissa = f.0 & 0b0000_0111;
        f32::from_bits((sign as u32) << 24 | ((exponent as u32>>3) + 111) << 23 | (mantissa as u32) << 20)
    }
}
impl From<f32> for f8 {
    fn from(f: f32) -> f8 {
        let bits = f.to_bits();
        let sign = (bits >> 24) & 0b1000_0000;
        let exponent = ((bits >> 23) & 0b0111_1111) - 111;
        let mantissa = (bits >> 20) & 0b0000_0111;
        f8((sign as u8) | ((exponent as u8)<<3) | (mantissa as u8))
    }
}
#[test]
fn test_f8() {
    println!("{:?}", f32::from(f8(0b0000_0001)));
    assert_eq!(f32::from(f8(0b0111_1000)), 0.5);
    assert_eq!(f32::from(f8(0b0111_0000)), 0.25);
    assert_eq!(f32::from(f8(0b0111_1001)), 0.5625);
    assert_eq!(f32::from(f8(0b0111_1010)), 0.625);
    assert_eq!(f32::from(f8(0b0111_1111)), 0.9375);

    assert_eq!(f8::from(0.5), f8(0b0111_1000));
    assert_eq!(f8::from(0.25), f8(0b0111_0000));
    assert_eq!(f8::from(0.5625), f8(0b0111_1001));
    assert_eq!(f8::from(0.625), f8(0b0111_1010));
    assert_eq!(f8::from(0.9375), f8(0b0111_1111));
}