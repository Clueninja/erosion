//! Erosion
//!  A library to plot Fourier Curves and Non-Continous Functions

pub mod prelude{
    use std::error::Error;
    pub trait Plottable{
        fn plot(self:&Self, output_file:&str, caption:&str, x_bound:(f32,f32), y_bound:(f32,f32), step:f32)->Result<(),Box<dyn Error>>;
    }

    pub trait Substitute{
        fn sub(self:&Self, x:f64)->f64;
    }
}


pub mod parser{
    use std::{iter::Peekable, str::Chars, error::Error};
    use super::prelude::*;

    /// a single node in a parser
    #[derive(Debug, Clone)]
    pub struct ParseNode{
        pub children:Vec<ParseNode>,
        pub entry:GrammarItem,
    }
    impl ParseNode{
        pub fn new()->ParseNode{
            ParseNode{
                children:Vec::new(),
                entry: GrammarItem::Paren,
            }
        }
    }

    /// a single Grammar Item for the parser
    #[derive(Debug, Clone)]
    pub enum GrammarItem{
        Paren,
        Float(f64),
        Var(char),
        Plus,
    }


    #[derive(Debug, Clone, PartialEq)]
    pub enum LexItem{
        Paren(char),
        Op(char),
        Num(f64),
        Var(char),
    }
    /// accepts a mutable refernence to an iterator, returns number of all the following digits
    pub fn char_as_num(it: &mut Peekable<Chars>)->Result<LexItem, Box<dyn Error>>{
        let mut astr = String::new();
        while let Some(&c) = &it.peek(){
            match  c{
                '0'..='9'=>{
                    astr.push_str(&c.to_string());
                    it.next();
                }
                _=>return Ok(LexItem::Num(astr.parse::<f64>().unwrap()))
            }
        }
        Ok(LexItem::Num(astr.parse::<f64>().unwrap()))
        
    }
    /// Accepts a string, returns a Vec of LexItems
    fn lex(input: &String)->Result<Vec<LexItem>, String>{
        let mut result = Vec::new();
        let mut it = input.chars().peekable();
        while let Some(&c) = it.peek(){
            match c{
                '('|')'=>{
                    result.push(LexItem::Paren(c));
                    it.next();
                }
                '+'|'-'=>{
                    result.push(LexItem::Op(c));
                    it.next();
                }
                ' '=>{
                    it.next();
                }

                '0'..='9'=>{
                    result.push(char_as_num(&mut it).unwrap());
                    it.next();
                }
                _=>{
                    return Err(format!("Unexpected Item in the Lexer: {}", c))
                }
            }
        }
        Ok(result)
    }
    /// accepts a string, returns a ParseNode
    pub fn parse(input:&String)->Result<ParseNode, String>{
        let tokens = lex(input)?;
        
        

        Ok(ParseNode::new())

    }

}


pub mod curves{
    use plotters::prelude::*;

    use super::prelude::{Plottable, Substitute};
    use std::error::Error;


    use std::io;
    use std::io::prelude::*;
    use std::fs::File;

    /// A sine curve in the form 
    /// mag*sine(fre*x + phase)
    pub struct Curve{
        pub mag: f64,
        pub fre:f64,
        pub phase:f64
    }

    impl Curve{
        pub fn new(mag:f64, fre:f64, phase:f64)->Self{
            Self{mag, fre, phase}
        }
    } // impl Curve
    /// sub into Curve a value for theta
    impl Substitute for Curve{
        fn sub(self:&Self, x:f64)->f64{
            self.mag*(self.fre*x+self.phase).sin()
        }
    }
    pub struct FourierCurve{
        curves: Vec<Curve>,
    }
    impl FourierCurve{
        pub fn new()->Self{
            FourierCurve{curves:Vec::new()}
        }
        /// push a curve into the Fourier Curve
        pub fn push(self:& mut Self, curve: Curve){
            self.curves.push(curve);
        }
        /// push components of a Curve into the Fourier Curve
        pub fn push_curve(self:&mut Self, mag:f64, fre:f64, phase:f64){
            self.push(Curve::new(mag, fre, phase));
        }
        
    } // impl FourierCourve
    impl Into<Vec<f64>> for FourierCurve{
        fn into(self) -> Vec<f64> {
            let mut result:Vec<f64> = Vec::new();
            let x_axis = (-3.14..3.14).step(0.01);
            for val in x_axis.values(){
                result.push(self.sub(val));
            }
            // read up to 10 bytes
            result
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
            chart.draw_series(LineSeries::new(x_axis.values().map(|x| (x, self.sub(x as f64) as f32)), &RED))?;
            //.label("Curve")
            //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            //chart.configure_series_labels().border_style(&BLACK).draw()?;

            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
            println!("Result has been saved to {}", output_file);
            Ok(())
    
        }
    }// impl Plottble for FourierCurve
    impl Substitute for FourierCurve{
        fn sub(self:&Self, x:f64)->f64{
            let mut sum = 0.0;
            for c in &self.curves{
                sum= sum + c.sub(x);
            }
            sum
        }
    }
        

}// pub mod curves

pub mod functions{
    use plotters::prelude::*;
    use super::prelude::*;

    use std::error::Error;

    /// A Function is a list of Bounded Polynomials
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
            chart.draw_series(LineSeries::new(x_axis.values().map(|x| (x, self.sub(x as f64) as f32)), &RED))?;
            //.label("Curve")
            //.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            //chart.configure_series_labels().border_style(&BLACK).draw()?;

            root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
            println!("Result has been saved to {}", output_file);
            Ok(())
        }
    }
    impl Substitute for Function{
        fn sub(self:&Self, x:f64) ->f64 {
            let mut ret = 0.;
            for t in &self.funcs{
                ret +=t.sub(x);
            }
            ret
        }
    }

    /// A bounded polynomial contains a polynomial and bounds in which it is defined
    pub struct BoundedPolynomial{
        pub poly:Polynomial,
        pub bounds:(f64, f64),
    }
    impl From<Polynomial> for BoundedPolynomial{
        fn from(poly: Polynomial) -> Self {
            Self{poly, bounds:(0.,10.)}
        }
    }

    

    impl Substitute for BoundedPolynomial{
        fn sub(self:&Self, x:f64) ->f64 {
            if self.bounds.0<x && self.bounds.1>x{
                return self.poly.sub(x);
            }
            0.
        }
    }

    /// contains a list of terms
    pub struct Polynomial{
        pub terms:Vec<Term>,
    }
    impl Polynomial{
        pub fn new()->Self{
            Self{terms: Vec::new()}
        }
    }

    impl Substitute for Polynomial{
        fn sub(self:&Self, x:f64) ->f64 {
            let mut ret = 0.;
            for t in &self.terms{
                ret +=t.sub(x);
            }
            ret
        }
    }

    /// A term is in the form
    /// (coef * x ^ pow)
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
        fn sub(self:&Self, x:f64)->f64{
            self.coef* x.powf(self.pow)
        }
    }
    
}





#[cfg(test)]
mod tests{
    use crate::{prelude::*, functions::*, curves::*};
    use std::error::Error;


    #[test]
    fn first(){
        let mut curve = FourierCurve::new();
        curve.push(Curve::new(1.,1., 0.));
        curve.push(Curve::new(0.5,3., 0.));
        curve.push(Curve::new(1./4.,5., 0.));
        curve.push(Curve::new(1./8., 7., 0.));
        
        curve.plot("plotters-doc-data/first_test.png","test 1", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


    }
    #[test]
    fn square(){
        let mut curve = FourierCurve::new();
        curve.push(Curve::new(4.0,1., 0.0));
        curve.push(Curve::new(4./3.,3.0, 0.0));
        curve.push(Curve::new(4./5.,5.0,0.0));
        curve.push(Curve::new(4./7.,7., 0.0));
       
        curve.plot("plotters-doc-data/square.png","square", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


    }
    #[test]
    fn saw_tooth(){
        let mut curve = FourierCurve::new();
        curve.push(Curve::new(-2.,1., 0.));
        curve.push(Curve::new(1.,2., 0.));
        curve.push(Curve::new(-2./3.,3., 0.));
        curve.push(Curve::new(0.5, 4., 0.));


        curve.plot("plotters-doc-data/saw_tooth.png","Saw Tooth", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


    }
    


    #[test]
    fn test_functions(){
        let mut curve = Function::new();
        curve.push(BoundedPolynomial{
            poly:Polynomial{
                terms:vec![Term::new(1., 1.), Term::new(2., 2.), Term::new(3., 3.)]
            }, 
            bounds:(-1000., 1000.),
        });
        curve.plot("plotters-doc-data/test_func.png", "Test Function", (-100.,100.), (-1000.,1000.), 0.01);
    }

    

    #[test]
    fn test_char_to_num(){
        use super::parser::char_as_num;
        let result = "a123a".to_string();
        let mut it = result.chars().peekable();
        it.next();
        let aint = char_as_num(&mut it).unwrap();
        assert_eq!(LexItem::Num(123.), aint);
        assert_eq!(it.peek(), Some(&'a'));


    }

}// mod tests