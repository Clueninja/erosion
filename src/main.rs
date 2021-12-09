
use curves::curves::{Curve, FourierCurve};
fn main(){
    let mut curve = FourierCurve::new();
    curve.push(Curve::new(1.0,1.0));
    curve.plot("plotters-doc-data/main.png", "Fourier Curve for Main", (-3.4, 3.4), (-1.2,1.2), 0.01).unwrap();
}
