//! Erosion
//!  A library to plot Fourier Curves and Non-Continous Functions

pub mod prelude{
    use std::error::Error;
    pub trait Plottable{
        fn plot(self:&Self, output_file:&str, caption:&str, x_bound:(f32,f32), y_bound:(f32,f32), step:f32)->Result<(),Box<dyn Error>>;
    }

    pub trait Substitute{
        fn substitute(self:&Self, x:f64)->f64;
    }

    pub trait Ordinal{
        fn ord(self:&Self)->f64;
    }
    pub trait Bounded{
        fn in_bounds(self:&Self, x:f64)->bool;
    }

    pub trait Calculus{
        fn integrate(self: &Self)-> Self;
        fn differenciate(self: &Self)->Self;
    }
}



pub mod curves{
    use plotters::prelude::*;
    use super::prelude::*;
    use std::{error::Error, f64::consts::PI, ops::{Add, Mul, Div, Sub, AddAssign, SubAssign, MulAssign, DivAssign}};
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Type{
        Sine,
        Cosine,
    }
    impl Type{
        pub fn inv(self:&Self)->Type{
            match self{
                Type::Sine=>Type::Cosine,
                Type::Cosine=>Type::Sine,
            }
        }
    }
    
    impl Add<f64> for Type{
        type Output = f64;
        fn add(self, rhs: f64) -> Self::Output {
            match self{
                Type::Sine=>rhs,
                Type::Cosine=>rhs+PI/2.,
            }
        }
    }

    /// A sine curve in the form 
    /// mag * sine ( fre * ( x + phase ) )
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Curve{
        pub mag : f64,
        pub fre : f64,
        pub phase : Type,
    }

    impl Curve{
        pub fn new(mag:f64, fre:f64, phase:Type)->Self{
            Self{mag, fre, phase}
        }
    } // impl Curve
    /// sub into Curve a value for theta
    impl Substitute for Curve{
        fn substitute(self:&Self, x:f64)->f64{
            self.mag*(self.fre*(self.phase + x)).sin()
        }
    }
    impl Calculus for Curve{
        fn integrate(self: &Self) -> Self {
            match self.phase{
                Type::Sine=>Self{
                    mag: self.mag/self.fre * -1.,
                    fre: self.fre,
                    phase: Type::Cosine,
                },
                Type::Cosine=>Self{
                    mag: self.mag/self.fre,
                    fre: self.fre,
                    phase: Type::Sine,
                }
            }
        }
        fn differenciate(self: &Self) ->Self {
            match self.phase{
                Type::Sine=>Self{
                    mag: self.mag*self.fre,
                    fre: self.fre,
                    phase: Type::Cosine,
                },
                Type::Cosine=>Self{
                    mag: self.mag*self.fre * -1.,
                    fre: self.fre,
                    phase: Type::Sine,
                }
            }
        }
    }
    impl AddAssign for Curve{
        fn add_assign(&mut self, rhs: Self) {
            if let Some(c) = *self + rhs{
                *self = c;
            }
        }
    }
    impl SubAssign for Curve{
        fn sub_assign(&mut self, rhs: Self) {
            if let Some(c) = *self - rhs{
                *self = c;
            }
        }
    }
    impl MulAssign<f64> for Curve{
        fn mul_assign(&mut self, rhs: f64) {
            *self = *self * rhs;
        }
    }
    impl DivAssign<f64> for Curve{
        fn div_assign(&mut self, rhs: f64) {
            *self = *self / rhs;
        }
    }
    impl Add<Curve> for Curve{
        type Output = Option<Curve>;
        fn add(self, rhs: Self) -> Self::Output {
            if self.fre == rhs.fre && self.phase == rhs.phase{
                return Some(
                    Curve::new(
                        self.mag + rhs.mag,
                        self.fre,
                        self.phase
                    )
                )
            }
            None
        }
    }
    impl Sub<Curve> for Curve{
        type Output = Option<Curve>;
        fn sub(self, rhs: Self) -> Self::Output {
            if self.fre == rhs.fre && self.phase == rhs.phase{
                return Some(
                    Curve::new(
                        self.mag - rhs.mag,
                        self.fre,
                        self.phase
                    )
                )
            }
            None
        }
    }
    impl Mul<f64> for Curve{
        type Output = Curve;
        fn mul(self, rhs: f64) -> Self::Output {
            Self::Output{
                mag: self.mag*rhs,
                fre:self.fre,
                phase: self.phase,
            }
        }
    }
    impl Div<f64> for Curve{
        type Output = Curve;
        fn div(self, rhs:f64) -> Self::Output{
            Self::Output{
                mag : self.mag/rhs,
                fre: self.fre,
                phase : self.phase
            }
        }
    }




    #[derive(Debug, Clone)]
    pub struct FourierCurve{
        pub curves: Vec<Curve>,
    }
    impl FourierCurve{
        pub fn new()->Self{
            FourierCurve{
                curves:Vec::new(),
            }
        }
        /// push a curve into the Fourier Curve
        pub fn push(self:& mut Self, curve: Curve){
            self.curves.push(curve);
        }
        /// push components of a Curve into the Fourier Curve
        pub fn push_curve(self:&mut Self, mag:f64, fre:f64, phase:Type){
            self.push(Curve::new(mag, fre, phase));
        }
        
    } // impl FourierCourve
    impl Into<Vec<f64>> for FourierCurve{
        fn into(self) -> Vec<f64> {
            let mut result:Vec<f64> = Vec::new();
            let x_axis = (-3.14..3.14).step(0.01);
            for val in x_axis.values(){
                result.push(self.substitute(val));
            }
            // read up to 10 bytes
            result
        }
    }
    impl Calculus for FourierCurve{
        fn integrate(self: &Self) -> Self {
            let mut f = FourierCurve::new();
            for c in &self.curves{
                f = f + c.integrate();
            }
            f

        }
        fn differenciate(self: &Self) ->Self {
            let mut f = FourierCurve::new();
            for c in &self.curves{
                f = f + c.differenciate();
            }
            f
        }
    }

    impl Plottable for FourierCurve{
    
        fn plot(self:&Self, output_file:&str, caption:&str, x_bound:(f32,f32), y_bound:(f32,f32), step:f32)->Result<(),Box<dyn Error>>{
            let root = BitMapBackend::new(output_file,(640,480)).into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(35)
                .y_label_area_size(40)
                .margin(5)
                .caption(caption, ("sans-serif", 50.0))
                .build_cartesian_2d(x_bound.0..x_bound.1, y_bound.0..y_bound.1)?;

            chart.configure_mesh()
                .x_labels(20)
                .y_labels(10)
                .disable_mesh()
                .x_label_formatter(&|v| format!("{:.1}", v))
                .y_label_formatter(&|v| format!("{:.1}", v))
                .draw()?;
                
            let x_axis = (x_bound.0..x_bound.1).step(step);
            chart.draw_series(LineSeries::new(x_axis.values().map(|x| (x, self.substitute(x as f64) as f32)), &RED))?;
            //.label("Curve")
            //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            //chart.configure_series_labels().border_style(&BLACK).draw()?;

            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
            println!("Result has been saved to {}", output_file);
            Ok(())
    
        }
    }// impl Plotable for FourierCurve
    impl Substitute for FourierCurve{
        fn substitute(self:&Self, x:f64)->f64{
            let mut sum = 0.0;
            for c in &self.curves{
                sum= sum + c.substitute(x);
            }
            sum
        }
    }
    impl PartialEq for FourierCurve{
        fn eq(&self, other: &Self) -> bool {
            for a in &self.curves{
                if !other.curves.contains(a){
                    return false
                }
            }
            true
        }
    }
    impl Mul<f64> for FourierCurve{
        type Output = FourierCurve;
        fn mul(self, rhs: f64) -> Self::Output {
            let mut f = self;
            for a in f.curves.iter_mut(){
                *a =  *a * rhs;
            }
            f
        }
    }
    
    impl Add<FourierCurve> for FourierCurve{
        type Output = FourierCurve;
        fn add(self, rhs: FourierCurve) -> Self::Output {
            let mut f = self;
            for c in rhs.curves{
                f = f + c;
            }
            f
        }
    }
    impl Add<Curve> for FourierCurve{
        type Output = FourierCurve;
        fn add(self, rhs: Curve) -> Self::Output {
            let mut f = self;
            let mut is_added = false;
            for c in f.curves.iter_mut(){
                match *c + rhs{
                    Some(curve)=>{
                        *c = curve;
                        is_added = true;
                    },
                    None=>{},
                }
            }
            if !is_added{
                f.curves.push(rhs);
            }
            f
        }
    }
    impl Sub<Curve> for FourierCurve{
        type Output = FourierCurve;
        fn sub(self, rhs: Curve) -> Self::Output {
            self + rhs * -1.
        }
    }
    impl Sub<FourierCurve> for FourierCurve{
        type Output = FourierCurve;
        fn sub(self, rhs: FourierCurve) -> Self::Output {
            self + rhs * -1.
        }
    }
        

}// pub mod curves

pub mod functions{
    use plotters::prelude::*;
    use super::prelude::*;

    use std::{error::Error, ops::{Sub, Add, Mul, Div}};

    /// A Function is a list of Bounded Polynomials
    #[derive(Debug, Clone, PartialEq)]
    pub struct Function{
        pub funcs:Vec<BoundedPolynomial>,
    }
    impl Function{
        pub fn new()->Self{
            Self{funcs:Vec::new()}
        }
        // TODO: check whether bounds overlap
        pub fn push(self:&mut Self, b_poly:BoundedPolynomial){
            self.funcs.push(b_poly);
        }
    }
    impl Bounded for Function{
        fn in_bounds(self:&Self, x:f64) ->bool {
            for p in &self.funcs{
                if p.in_bounds(x){
                    return true
                }
            }
            false
        }
    }
    impl Calculus for Function{
        fn integrate(self: &Self) -> Self {
            let mut f = Function::new();
            for bp in &self.funcs{
                f.push(bp.integrate());
            }
            f
        }
        fn differenciate(self: &Self) ->Self {
            let mut f = Function::new();
            for bp in &self.funcs{
                f.push(bp.differenciate());
            }
            f
        }
    }
    impl Ordinal for Function{
        fn ord(self:&Self) ->f64 {
            let mut max = f64::NEG_INFINITY;
            for f in & self.funcs{
                if f.ord()>max{
                    max = f.ord();
                }
            }
            max
        }
    }
    impl Plottable for Function{
        fn plot(self:&Self, output_file:&str, caption:&str, x_bound:(f32,f32), y_bound:(f32,f32), step:f32) ->Result<(),Box<dyn Error>> {
            let root = BitMapBackend::new(output_file,(1080,920)).into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .x_label_area_size(35)
                .y_label_area_size(40)
                .margin(5)
                .caption(caption, ("sans-serif", 50.0))
                .build_cartesian_2d(x_bound.0..x_bound.1, y_bound.0..y_bound.1)?;

            chart.configure_mesh()
                .x_labels(20)
                .y_labels(10)
                .disable_mesh()
                .x_label_formatter(&|v| format!("{:.1}", v))
                .y_label_formatter(&|v| format!("{:.1}", v))
                .draw()?;
                
            let x_axis = (x_bound.0..x_bound.1).step(step);
            chart.draw_series(LineSeries::new(x_axis.values().map(|x| (x, self.substitute(x as f64) as f32)), &RED))?;
            //.label("Curve")
            //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            //chart.configure_series_labels().border_style(&BLACK).draw()?;

            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
            println!("Result has been saved to {}", output_file);
            Ok(())
        }
    }
    impl Substitute for Function{
        fn substitute(self:&Self, x:f64) ->f64 {
            let mut ret = 0.;
            for t in &self.funcs{
                ret +=t.substitute(x);
            }
            ret
        }
    }

    /// A bounded polynomial contains a polynomial and bounds in which it is defined
    #[derive(Debug, Clone, PartialEq)]
    pub struct BoundedPolynomial{
        pub poly:Polynomial,
        pub bounds:(f64, f64),
    }
    impl BoundedPolynomial{
        pub fn bounds_mut(self:&mut Self)-> &mut (f64, f64){
            &mut self.bounds
        }
    }
    impl Bounded for BoundedPolynomial{
        fn in_bounds(self:&Self, x:f64)->bool{
            self.bounds.0<=x && x< self.bounds.1
        }
    }
    impl From<Polynomial> for BoundedPolynomial{
        fn from(poly: Polynomial) -> Self {
            Self{poly, bounds:(0.,10.)}
        }
    }
    impl Ordinal for BoundedPolynomial{
        fn ord(self:&Self) ->f64 {
            self.poly.ord()
        }
    }
    impl Calculus for BoundedPolynomial{
        fn differenciate(self: &Self) ->Self {
            Self{
                poly : self.poly.differenciate(),
                bounds: self.bounds,
            }
        }
        fn integrate(self: &Self) -> Self {
            Self{
                poly : self.poly.integrate(),
                bounds: self.bounds,
            }
        }
    }
    impl Substitute for BoundedPolynomial{
        fn substitute(self:&Self, x:f64) ->f64 {
            if self.bounds.0<x && self.bounds.1>x{
                return self.poly.substitute(x);
            }
            0.
        }
    }
    
    #[derive(Debug, Clone)]
    /// contains a list of terms
    pub struct Polynomial{
        pub terms:Vec<Term>,
    }
    impl Polynomial{
        pub fn new()->Self{
            Self{terms: Vec::new()}
        }
    }
    impl Ordinal for Polynomial{
        fn ord(self:&Self) ->f64 {
            let mut max = f64::NEG_INFINITY;
            for t in &self.terms{
                if t.pow>max{
                    max = t.pow;
                }
            }
            max
        }
    }
    impl Calculus for Polynomial{
        fn integrate(self: &Self) -> Self {
            let mut p = Polynomial::new();
            for t in &self.terms{
                p = p + t.integrate();
            }
            p
        }
        fn differenciate(self: &Self) ->Self {
            let mut p = Polynomial::new();
            for t in &self.terms{
                p = p + t.differenciate();
            }
            p
        }
    }

    impl Substitute for Polynomial{
        fn substitute(self:&Self, x:f64) ->f64 {
            let mut ret = 0.;
            for t in &self.terms{
                ret +=t.substitute(x);
            }
            ret
        }
    }
    impl PartialEq for Polynomial{
        fn eq(&self, other: &Self) -> bool {
            for a in &self.terms{
                if !other.terms.contains(a){
                    return false
                }
            }
            true
        }
    }
    
    impl Add<Polynomial> for Polynomial{
        type Output = Polynomial;
        fn add(self, rhs: Polynomial) -> Self::Output {
            let mut poly = self;
            for t in rhs.terms{
                poly = poly + t;
            }
            poly
        }
    }
    impl Add<Term> for Polynomial{
        type Output = Polynomial;
        fn add(self, rhs: Term) -> Self::Output {
            let mut poly = self;
            let mut is_added = false;
            for t in poly.terms.iter_mut(){
                match *t + rhs{
                    Some(term)=>{
                        *t = term;
                        is_added = true;
                    },
                    None=>{},
                }
            }
            if !is_added{
                poly.terms.push(rhs);
            }
            poly
        }
    }

    /// A term is in the form
    /// (coef * x ^ pow)
    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub struct Term{
        pub coef:f64,
        pub pow:f64
    }
    impl Term{
        pub fn new(coef:f64, pow:f64)->Self{
            Self{pow, coef}
        }
    }
    impl Substitute for Term{
        fn substitute(self:&Self, x:f64)->f64{
            self.coef* x.powf(self.pow)
        }
    }
    impl Calculus for Term{
        fn differenciate(self: &Self) ->Self {
            Self{
                coef: self.coef*self.pow,
                pow: self.pow-1.
            }
        }
        fn integrate(self: &Self) -> Self {
            Self{
                coef: self.coef/(self.pow+1.),
                pow: self.pow+1.,
            }
        }
    }
    /// Subtracting and Adding Terms can only work iff the pow is the same
    impl Sub for Term{
        type Output = Option<Term>;
        fn sub(self, rhs: Self) -> Self::Output {
            if self.pow == rhs.pow{
                return Some(Self{
                    coef: self.coef-rhs.coef,
                    pow:self.pow,
                })
            }
            None
        }
    }
    impl Add for Term{
        type Output = Option<Term>;
        fn add(self, rhs: Self) -> Self::Output {
            if self.pow == rhs.pow{
                return Some(Self{
                    coef: self.coef+rhs.coef,
                    pow:self.pow,
                })
            }
            None
        }
    }
    impl Mul<Term> for Term{
        type Output = Term;
        fn mul(self, rhs: Term) -> Self::Output {
            Self::Output{
                coef:self.coef*rhs.coef,
                pow:self.pow+rhs.pow,
            }
        }
    }
    impl Div<Term> for Term{
        type Output = Term;
        fn div(self, rhs: Term) -> Self::Output {
            Self::Output{
                coef:self.coef/rhs.coef,
                pow:self.pow-rhs.pow,
            }
        }
    }
    impl Mul<f64> for Term{
        type Output = Term;
        fn mul(self, rhs: f64) -> Self::Output {
            Self::Output{
                coef: self.coef*rhs,
                pow: self.pow,
            }
        }
    }
    impl Div<f64> for Term{
        type Output = Term;
        fn div(self, rhs: f64) -> Self::Output {
            Self::Output{
                coef: self.coef/rhs,
                pow: self.pow,
            }
        }
    }
    
}





#[cfg(test)]
mod tests{
    use crate::{prelude::*, functions::*, curves::{*, Type::*}};

    mod test_plots{
        use super::*;

        #[test]
        fn first(){
            let mut curve = FourierCurve::new();
            curve.push(Curve::new(1.,1., Sine));
            curve.push(Curve::new(0.5,3., Sine));
            curve.push(Curve::new(1./4.,5., Sine));
            curve.push(Curve::new(1./8., 7., Sine));
            
            curve.plot("plotters-doc-data/first_test.png","test 1", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


        }
        #[test]
        fn square(){
            let mut curve = FourierCurve::new();
            curve.push(Curve::new(4.0,1., Sine));
            curve.push(Curve::new(4./3.,3.0, Sine));
            curve.push(Curve::new(4./5.,5.0,Sine));
            curve.push(Curve::new(4./7.,7., Sine));
        
            curve.plot("plotters-doc-data/square.png","square", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


        }
        #[test]
        fn saw_tooth(){
            let mut curve = FourierCurve::new();
            curve.push(Curve::new(-2.,1., Sine));
            curve.push(Curve::new(1.,2., Sine));
            curve.push(Curve::new(-2./3.,3., Sine));
            curve.push(Curve::new(0.5, 4., Sine));


            curve.plot("plotters-doc-data/saw_tooth.png","Saw Tooth", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


        }
    } // mod test_plots
    mod test_function{
        use super::*;
        #[test]
        fn test_functions(){
            let mut curve = Function::new();
            curve.push(BoundedPolynomial{
                poly:Polynomial{
                    terms:vec![Term::new(1., 1.), Term::new(2., 2.), Term::new(3., 3.)]
                }, 
                bounds:(-1000., 1000.),
            });
            curve.plot("plotters-doc-data/test_func.png", "Test Function", (-100.,100.), (-1000.,1000.), 0.01).unwrap();
        }
        #[test]
        fn function_binary(){
            // Polynomial tests
            let mut p = Polynomial::new();

            // add term to polynomial
            p = p + Term::new(1., 1.);
            assert_eq!(
                p, 
                Polynomial{
                    terms:vec!(
                        Term::new(1., 1.),
                    )
                }
            );
            // add another term with same pow
            p = p + Term::new(2., 1.);
            assert_eq!(
                p, 
                Polynomial{
                    terms:vec!(Term::new(3., 1.))
                }
            );
            // add another term with a differnent pow
            p = p + Term::new(2., 2.) + Term::new(3., 3.);
            assert_eq!(p, 
                Polynomial{
                    terms:vec!(
                        Term::new(3., 3.),
                        Term::new(3., 1.),
                        Term::new(2.,2.),
                    )
                }
            );
            // Term tests
            let t = Term::new(1., 1.);
            
            // Term op Term tests
            // add
            assert_eq!(t + Term::new(2., 1.), Some(Term::new(3., 1.)));
            assert_eq! (t + Term::new(2., 2.), None);
            // sub
            assert_eq!(t-Term::new(0.5, 1.), Some(Term::new(0.5, 1.)));
            assert_eq!(t-Term::new(2., 1.), Some(Term::new(-1., 1.)));

            // Term op f64 tests
            // mul
            assert_eq!(t * 2. , Term::new(2., 1.));
            // div
            assert_eq!(t / 5., Term::new(1./5., 1.));

            // Term op Term tests
            // mul
            assert_eq!(t * Term::new(2., 1.), Term::new(2., 2.));
            assert_eq!(t * Term::new(3., 2.), Term::new(3., 3.));
            // div
            assert_eq!(t / Term::new(3., 2.), Term::new(1./3., -1.));
        }
        #[test]
        fn test_bounds(){
            let mut bp = BoundedPolynomial::from(Polynomial{
                terms: vec!(
                    Term::new(1.,2.),
                    Term::new(2.,3.),
                )
            });
            assert!(bp.in_bounds(-1.) == false);
            assert!(bp.in_bounds(5.) == true);
            assert!(bp.in_bounds(11.) == false);

            let b = bp.bounds_mut();
            b.0 = -10.0;
            
            assert!(bp.in_bounds(-1.) == true);
            assert!(bp.in_bounds(5.) == true);
            assert!(bp.in_bounds(11.) == false);

            assert!(bp.in_bounds(-10.) == true);
        
            assert!(bp.in_bounds(10.) == false);
        }
        
        #[test]
        fn test_calculus(){
            let t = Term::new(2., 3.);
            assert_eq!(
                t.integrate(),
                Term::new(2./4., 4.)
            );
            assert_eq!(
                t.differenciate(),
                Term::new(2.*3., 2.)
            );


            let mut p = Polynomial{
                terms : vec!(
                    Term::new(2., 3.)
                )
            };
            assert_eq!(
                p.integrate(),
                Polynomial{
                    terms : vec!(
                        Term::new(2., 3.).integrate(),
                    )
                }
            );
            p = p + Term::new(3., 5.);

            assert_eq!(
                p.integrate(),
                Polynomial{
                    terms : vec!(
                        Term::new(2., 3.).integrate(),
                        Term::new(3., 5.).integrate(),
                    )
                }
            );

            let mut f = Function::new();
            f.push(
                BoundedPolynomial{
                    poly: Polynomial{
                        terms: vec!(
                            Term::new(2., 3.),
                        )
                    }, 
                    bounds: (0., 10.),
                }
            );

            assert_eq!(
                f.integrate(),
                Function{
                    funcs: vec!(
                        BoundedPolynomial{
                            poly: Polynomial{
                                terms: vec!(
                                    Term::new(2., 3.).integrate(),
                                )
                            }, 
                            bounds: (0., 10.),
                        }
                    )
                }
            );

            f.push(
                BoundedPolynomial{
                    poly: Polynomial{
                        terms: vec!(
                            Term::new(2., 4.),
                        )
                    }, 
                    bounds: (0., 10.),
                }
            );

            assert_eq!(
                f.integrate(),
                Function{
                    funcs: vec!(
                        BoundedPolynomial{
                            poly: Polynomial{
                                terms: vec!(
                                    Term::new(2., 3.).integrate(),
                                )
                            }, 
                            bounds: (0., 10.),
                        },
                        BoundedPolynomial{
                            poly: Polynomial{
                                terms: vec!(
                                    Term::new(2., 4.).integrate(),
                                )
                            }, 
                            bounds: (0., 10.),
                        }
                    )
                }
            );
        }
    } // mod test_function
    mod test_curves{
        use super::*;
        #[test]
        fn curves_binary(){
            let c = Curve::new(1., 1.,  Sine);
            assert_eq!(
                c + Curve::new(  2.,  1.,  Sine) ,
                Some(Curve::new(3.,  1.,  Sine))
            );

            assert_eq!(
                c + Curve::new(  2.,  1.,  Cosine) , 
                None
            );

            assert_eq!(
                c + Curve::new(  2.,  2.,  Sine) , 
                None
            );

            assert_eq!(
                c + Curve::new(  2.,  2.,  Cosine) , 
                None
            );

            let mut f = FourierCurve{
                curves: vec!(
                    Curve::new(1., 1., Sine),
                )
            };

            f = f + Curve::new(2., 1., Sine);
            assert_eq!(
                f ,
                FourierCurve{
                    curves: vec!(
                        Curve::new(3., 1., Sine),
                    )
                }
            );
            f = f + Curve::new(2., 2., Sine);
            assert_eq!(
                f,
                FourierCurve{
                    curves: vec!(
                        Curve::new(3., 1., Sine),
                        Curve::new(2., 2., Sine),
                    )
                }
            );
        }// fn curves_binary

        #[test]
        fn curves_calculus(){
            let c = Curve::new(2., 3., Sine);
            // Sine integrates to - Cosine
            assert_eq!(
                c.integrate(),
                Curve::new(-2./3., 3., Cosine)
            );

            // Sine differentiates to Cosine
            assert_eq!(
                c.differenciate(),
                Curve::new(2.*3., 3., Cosine)
            );

            let mut f = FourierCurve{
                curves:vec!(
                    Curve::new(2., 3., Sine)
                )
            };
            assert_eq!(
                f.integrate(),
                FourierCurve{
                    curves:vec!(
                        Curve::new(2., 3., Sine).integrate()
                    )
                }
            );

        }
    } // mod test_curves
}// mod tests