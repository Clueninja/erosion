//! Erosion
//! 'erosion' is a library to plot Fourier Curves and Non-Continous Functions

pub mod parser{
    use std::{iter::Peekable, str::Chars, error::Error};

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


    #[derive(Debug, Clone)]
    pub enum LexItem{
        Paren(char),
        Plus,
        Num(f64),
        Var(char),
    }
    /// accepts a mutable refernence to an iterator, returns number of all the following digits
    fn char_as_num(it: &mut Peekable<Chars>)->Result<LexItem, Box<dyn Error>>{
        let mut astr = String::new();
        loop{
            if let Some(&c) = &it.peek(){
                match  c{
                    '0'..='9'=>{
                        astr.push_str(&c.to_string());
                        it.next();
                    }
                    _=>return Ok(LexItem::Num(astr.parse::<f64>().unwrap()))
                }
            }
        }
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
                '+'=>{
                    result.push(LexItem::Plus);
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

    use super::{Plottable, Substitute};
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

        /// outputs the Fourier Curve as a binary file
        pub fn as_bytes(self:&Self, file_name:&str, x_bound:(f64,f64), step:f64, length : usize)-> io::Result<()> {
            let mut f = File::create(file_name)?;
            let mut buffer = [0; 1024*1024];
            for index in 0..length{
                let byte = 8.0*(self.sub(x_bound.0+step*(index as f64))+14.0);
                buffer[index] =byte as u8;
                //println!("{}", step*(index as f64));
            }
            // read up to 10 bytes
            let n = f.write(&mut buffer)?;
        
            println!("The bytes: {:?}", &buffer[..n]);
            Ok(())
        }
    } // impl FourierCourve
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

    use super::{Plottable, Substitute};
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

use std::error::Error;
pub trait Plottable{
    fn plot(self:&Self, output_file:&str, caption:&str, x_bound:(f32,f32), y_bound:(f32,f32), step:f32)->Result<(),Box<dyn Error>>;
}

pub trait Substitute{
    fn sub(self:&Self, x:f64)->f64;
}



#[cfg(test)]
mod tests{
    use crate::{Plottable, Substitute, functions::*, curves::{FourierCurve, Curve}};


    #[test]
    fn first(){
        let mut curve = FourierCurve::new();
        curve.push(Curve::new(5.0,2.0, 0.0));
        
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
    fn test_buffer(){
        let mut curve = FourierCurve::new();
        curve.push(Curve::new(10.0,0.1, 1.2));
        curve.push(Curve::new(2.0,1.0, 5.3));
        curve.push(Curve::new(0.5,2.0,3.2));
        curve.push(Curve::new(1.0,0.2,2.2));
        curve.push(Curve::new(3.0,3.0,0.2));
        curve.push(Curve::new(0.1,5.2, 5.0));

        let result = curve.as_bytes("plotters-doc-data/test_buffer", (0.0,200.0), (1.0/8000.0) as f64, 1024*1024);
        match result{
            Err(_)=>panic!("There was a file error"),
            Ok(())=>(),
        }

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

}// mod tests