use std::error::Error;

use curves::curves::{Curve, FourierCurve};
use plotters::prelude::*;
const OUT_FILE_NAME: &'static str = "plotters-doc-data/first.png";
fn main(){
    let mut curve = FourierCurve::new();
    curve.push(Curve::new(1.0,1.0));
    curve.plot("plotters-doc-data/main.png", "Fourier Curve for Main", (-3.4, 3.4), (-1.2,1.2), 0.01).unwrap();
}
