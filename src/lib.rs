pub mod curves{
    use plotters::prelude::*;
    use std::error::Error;

    pub struct Curve{
        pub mag: f64,
        pub fre:f64,
    }

    impl Curve{
        pub fn new(mag:f64, fre:f64)->Self{
            Self{mag, fre}
        }
        pub fn sub(self:&Self, x:f64)->f64{
            self.mag*(self.fre*x).sin()
        }
    } // impl Curve
    pub struct FourierCurve{
        curves: Vec<Curve>,
    }
    impl FourierCurve{
        pub fn new()->Self{
            FourierCurve{curves:Vec::new()}
        }
        pub fn push(self:& mut Self, curve: Curve){
            self.curves.push(curve);
        }
        pub fn push_curve(self:&mut Self, mag:f64, fre:f64){
            self.push(Curve::new(mag, fre));
        }
        pub fn sub(self:&Self, x:f64)->f64{
            let mut sum = 0.0;
            for c in &self.curves{
                sum= sum + c.sub(x);
            }
            sum
        }
        pub fn plot(self:&Self, output_file:&str, caption:&str, x_bound:(f32,f32), y_bound:(f32,f32), step:f32)->Result<(),Box<dyn Error>>{
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
    }// impl FourierCourves
}// mod curves

#[cfg(test)]
mod tests{
    use crate::curves::{FourierCurve, Curve};
    #[test]
    fn first(){
        let mut curve = FourierCurve::new();
        curve.push(Curve::new(5.0,2.0));
        
        curve.plot("plotters-doc-data/first_test.png","test 1", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


    }
    #[test]
    fn second(){
        let mut curve = FourierCurve::new();
        curve.push(Curve::new(10.0,0.1));
        curve.push(Curve::new(2.0,1.0));
        curve.push(Curve::new(0.5,2.0));
        curve.push(Curve::new(1.0,0.2));
        curve.push(Curve::new(3.0,3.0));
        curve.push(Curve::new(0.1,5.2));

        curve.plot("plotters-doc-data/second_test.png","test 2", (0.0,20.0), (-10.0, 20.0), 0.01).unwrap();


    }

}// mod tests