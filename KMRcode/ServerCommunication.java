package com.test.application;

import static com.kuka.roboticsAPI.motionModel.BasicMotions.lin;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.ptp;

import javax.inject.Inject;
import javax.inject.Named;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.json.simple.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import sun.nio.cs.StandardCharsets;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.IOException;

import kmr.io.EGI80Gripper;
import kmr.io.FlangeIOSignals;

import org.apache.commons.codec.binary.Base64;
import com.kuka.roboticsAPI.deviceModel.JointEnum;



import com.kuka.parameters.IParameterMap;
import com.kuka.common.honk.v1alpha1.HonkCommand;
import com.kuka.motion.IMotionContainer;
import com.kuka.nav.Location;
import com.kuka.nav.OrientationMode;
import com.kuka.nav.Pose;
import com.kuka.nav.data.LocationData;
import com.kuka.nav.honk.HonkAction;
import com.kuka.nav.line.VirtualLine;
import com.kuka.nav.line.VirtualLineMotion;
import com.kuka.nav.rel.RelativeMotion;
import com.kuka.nav.robot.LocalizeCommand;
import com.kuka.nav.robot.MobileRobot;
import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.executionModel.CommandInvalidException;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.geometricModel.ObjectFrame;
import com.kuka.task.ITaskLogger;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import com.vividsolutions.jts.geomgraph.Position;


public class _ServerCommunication extends RoboticsAPIApplication{
		private static final String IMAGE_PATH = "captured_image.jpg";
		
		private IParameterMap llk;
		
		@Inject
		private EGI80Gripper gripper;
	    
		@Inject
	    private LBR robot;
		private String operation = "";
		private boolean run = true;
		private String fail = "";
		private String motion = "";
		private double TX = 0;
		private double TY = 0;
		private double TZ = 0;
		private double TA = 0;
		private double TB = 0;
		private double TC = 0;
		private double A1 = 0;
		private double A2 = 0;
		private double A3 = 0;
		private double A4 = 0;
		private double A5 = 0;
		private double A6 = 0;
		private double A7 = 0;
		private double speed = 0.1;
		
		long initSpeed = 200000L;
		long initForce = 1L;
	    
	    private HttpServer server;
	    
		@Inject
		private FlangeIOSignals flange;

		@Inject
		@Named("KMR200_1")
		private MobileRobot kmp;
		
		@Inject
		private ITaskLogger log;
		
	    @Inject
	    private LocationData locationData;
		
		
	    @Override
	    public void initialize() {
	        try {
	            server = HttpServer.create(new InetSocketAddress(30000), 0);
	            server.createContext("/answerClient", new AnswerClientHandler());
	            server.createContext("/Count", new CountHandler());
	            server.createContext("/ArrowsMove", new ArrowsMove());
	            server.createContext("/GetPose", new GetPose());
	            server.createContext("/SetPose", new SetPose());
	            server.createContext("/HonkOn", new HonkOn());
	            server.createContext("/HonkOff", new HonkOff());
	            server.createContext("/GotoPosition", new GotoPosition());
	            server.createContext("/GetIIWAposition", new GetIIWAposition());
	            server.createContext("/GetIIWAJointsPosition", new GetIIWAJointsPosition());
	            server.createContext("/GotoJoint", new GotoJoint());
	            server.createContext("/GoHome", new GoHome());
	            server.createContext("/MoveToLocation", new MoveToLocation());
	            server.createContext("/CloseGripper", new CloseGripper());
	            server.createContext("/OpenGripper", new OpenGripper());
	            server.createContext("/InitGripper", new InitGripper());
	            server.createContext("/ReleaseObject", new ReleaseObject());
	            server.createContext("/GetGripperState", new GetGripperState());
	            server.createContext("/SetLED", new SetLED());

	            
	            
	            
	            

	            // server.createContext("/CaptureImage", new CaptureImage());
	            
	            
	            server.setExecutor(null);
	            server.start();
	            System.out.println("Server started on port 30000");
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	    }
	    
	    // ------------------------
	    
	    class SetLED implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "Switched";
				
				Map <String,String>parms = queryToMap(t.getRequestURI().getQuery());
				String color = parms.get("color");
				
				try{
					if(color == "blue"){
						flange.blueLED();
					}else if(color == "red"){
						flange.redLED();
					}else if(color == "green"){
						flange.greenLED();
					}else{
						response = "unknown color";
					}			
				}catch (Exception e){
					System.out.println("Exception is: "+e);
					response = "" + e;
				}
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}
	    
	    class GetGripperState implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "";
				
				try{
					JSONObject position= new JSONObject();   
					
					Long GripperPose = gripper.getPosition();
					
					position.put("position", GripperPose);

					response = position.toString();
					
				}catch (Exception e){
					System.out.println("Exception is: "+e);
					response = "" + e;
				}
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}
	    
	    
	    class ReleaseObject implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "released";
				
				try{
					gripper.releaseObject();
				}catch (Exception e){
					System.out.println("Exception is: "+e);
					response = "" + e;
				}
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}

	    
	    class InitGripper implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "initialized";
				
				try{
					gripper.initializeGripper();
					gripper.setSpeed(initSpeed);
					gripper.setForce(initForce);
				}catch (Exception e){
					System.out.println("Exception is: "+e);
					response = "" + e;
				}
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}

	    class OpenGripper implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "opened";
				
				try{
					gripper.openGripper();
				}catch (Exception e){
					System.out.println("Exception is: "+e);
					response = "" + e;
				}
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}
	    
	    
	    class CloseGripper implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "closed";
				Map <String,String>parms = queryToMap(t.getRequestURI().getQuery());
				Long force = Long.parseLong(parms.get("force"));
				
				try{
					gripper.gripWithForce(force);
				}catch (Exception e){
					System.out.println("Exception is: "+e);
					response = "" + e;
				}
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
		class GotoPosition implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "KK";
				
				System.out.println("Goto Cartesian Request");
				System.out.println(t.getRequestURI().getQuery());
			
				Map <String,String>parms = queryToMap(t.getRequestURI().getQuery());
				
				
				if (operation=="" || operation=="failed") {
					operation = "GotoCartesianposition";
					TX = Double.parseDouble(parms.get("x"));
					TY = Double.parseDouble(parms.get("y"));
					TZ = Double.parseDouble(parms.get("z"));
					TA = Double.parseDouble(parms.get("a"));
					TB = Double.parseDouble(parms.get("b"));
					TC = Double.parseDouble(parms.get("c"));
					speed = Double.parseDouble(parms.get("Speed"));
					motion = parms.get("Motion").toString().trim();

					response = "OK";
				}
				else{
					response = "busy";		
				}
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}
		
		class GotoJoint implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response;
				
				System.out.println("Goto Joint Request");
				System.out.println(t.getRequestURI().getQuery());
			
				Map <String,String>parms = queryToMap(t.getRequestURI().getQuery());
				
				if (operation=="" || operation=="failed") {
					operation = "GotoJointposition";
					A1 = Double.parseDouble(parms.get("A1"));
					A2 = Double.parseDouble(parms.get("A2"));
					A3 = Double.parseDouble(parms.get("A3"));
					A4 = Double.parseDouble(parms.get("A4"));
					A5 = Double.parseDouble(parms.get("A5"));
					A6 = Double.parseDouble(parms.get("A6"));
					A7 = Double.parseDouble(parms.get("A7"));
					speed = Double.parseDouble(parms.get("speed"));
					
					robot.move(ptp(new JointPosition(A1, A2, A3, A4, A5, A6, A7)).setJointVelocityRel(speed));
					
					response = "OK";
				}
				else
					response = "busy";			
				
				
				t.sendResponseHeaders(200, response.length());
				OutputStream os = t.getResponseBody();
				os.write(response.getBytes());
				os.close();
				
			}
		}
			
		class GetIIWAposition implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
				String response;
				
				ObjectFrame actual_pos = robot.getFlange();
				JSONObject position= new JSONObject();   
				Frame pos = actual_pos.copy();

				position.put("x", pos.getX());
				position.put("y", pos.getY());
				position.put("z", pos.getZ());
				position.put("A", pos.getAlphaRad());
				position.put("B", pos.getBetaRad());
				position.put("C", pos.getGammaRad());
				
				response = position.toString();
				
				System.out.println("" + position);
				
				t.sendResponseHeaders(200, response.length());
				OutputStream os = t.getResponseBody();
				os.write(response.getBytes());
				os.close();
			}
		}
		
		
		class GetIIWAJointsPosition implements HttpHandler {

			@Override
			public void handle(HttpExchange t) throws IOException {
				System.out.println("Getting the position.");
				String response;
				
				ObjectFrame actual_pos = robot.getFlange();
				JSONObject position= new JSONObject();   
				Frame pos = actual_pos.copy();

				// position.put("A1", pos.get("A1");
				JointPosition positionJoint;
				positionJoint = new JointPosition(7);
				
				// position.put("A1", positionJoint.get(1));
				// robot.getJointValue(JointEnum.J1);
				
				
				JointPosition currentJointPosition = robot.getCurrentJointPosition();
				double joint1Angle = currentJointPosition.get(JointEnum.J1); 
				joint1Angle = currentJointPosition.get(0);
				
				
				System.out.println("be for e.");
				for (int i = 0; i < 7; i++) {
					positionJoint.set(i, currentJointPosition.get(i));
					position.put("A" + (i + 1), currentJointPosition.get(i));
				}
				System.out.println("After for.");

				
				System.out.println("" + positionJoint);
				
				response = position.toString();
				
				t.sendResponseHeaders(200, response.length());
				OutputStream os = t.getResponseBody();
				os.write(response.getBytes());
				os.close();
			}
		}
	    
		
	    class GoHome implements HttpHandler {
	        @Override
	        public void handle(HttpExchange t) throws IOException {
	        	
	        	
	        	GotoJoint(0, 0, 0, 0, 0, 0, 0);
                
	        	String response = "home";
	        	
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	        }
	    }
	    
	    // -----------------------
	    
	    
	    class HonkOn implements HttpHandler {
	        @Override
	        public void handle(HttpExchange t) throws IOException {

	        	com.kuka.nav.honk.HonkCommand honk = com.kuka.nav.honk.HonkCommand.ON;
	        	HonkAction Honking = new HonkAction(honk);
	        	
	        	try{
		        	kmp.lock();
		        	kmp.execute(Honking);
		        }catch (Exception e){
		        	log.error("Something went wrong while trying to excecute navigation motions on the robot.", e);
				}finally{
					kmp.unlock();
					getLogger().info("KMR unlocked.");
				}
	        	String response = "on";
	        	
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	        }
	    }
	    
	    class HonkOff implements HttpHandler {
	        @Override
	        public void handle(HttpExchange t) throws IOException {

	        	com.kuka.nav.honk.HonkCommand honk = com.kuka.nav.honk.HonkCommand.OFF;
	        	HonkAction Honking = new HonkAction(honk);
	        	
	        	try{
		        	kmp.lock();
		        	kmp.execute(Honking);
		        }catch (Exception e){
		        	log.error("Something went wrong while trying to excecute navigation motions on the robot.", e);
				}finally{
					kmp.unlock();
					getLogger().info("KMR unlocked.");
				}
	        	String response = "off";
	        	
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	        }
	    }
	    
	    class SetPose implements HttpHandler {
	        @Override
	        public void handle(HttpExchange t) throws IOException {
			    Pose position = kmp.getPose();
		        System.out.println("Pose before setting: " + position);
		        
		        Pose SetPosition = new Pose(0, 0, 0);
		        
		        LocalizeCommand command = new LocalizeCommand(SetPosition);
		        
		        try{
		        	kmp.lock();
		        	kmp.execute(command);
		        }catch (Exception e){
		        	log.error("Something went wrong while trying to excecute navigation motions on the robot.", e);
				}finally{
					kmp.unlock();
					getLogger().info("KMR unlocked.");
				}
		        
		        
		        position = kmp.getPose();
		        System.out.println("Pose after setting: " + position);
		        
		        String response = "" + position;
		        
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	        }
	    }

	    class GetPose implements HttpHandler {
	        @Override
	        public void handle(HttpExchange t) throws IOException {
	            Pose position = kmp.getPose();
	            
	            // Create a JSON object with x, y, and theta
	            JSONObject poseJson = new JSONObject();
	            poseJson.put("x", position.getX());
	            poseJson.put("y", position.getY());
	            poseJson.put("theta", position.getTheta());

	            String response = poseJson.toString();
	            System.out.println(response);

	            // Send response
	            t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	        }
	    }

	    class AnswerClientHandler implements HttpHandler {
	        @Override
	        public void handle(HttpExchange t) throws IOException {
	            String response = "Hello from KUKA Robot Server!";
	            t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	        }
	    }
	    
	    
	    class MoveToLocation implements HttpHandler {

			@SuppressWarnings("null")
			@Override
			public void handle(HttpExchange t) throws IOException {
		
				String response = "BLBLBLBL";
				
				System.out.println(t.getRequestURI().getQuery());
				
				
				
				Map <String,String>parms = queryToMap(t.getRequestURI().getQuery());
				
				int TargetNumber = (int) Double.parseDouble(parms.get("TargetNumber"));
					
				
				
				Location targetLocation = locationData.get(TargetNumber);
				
	            if (targetLocation == null){
	                getLogger().error("Location not found");
	                return;
	            }
	            
	            System.out.println("Q1");
	            //VirtualLine Line = new VirtualLine(kmp.getPose().toPosition(), targetLocation.getPose().toPosition()); 
	            


	            com.kuka.nav.Position currentLocation = kmp.getPose().toPosition();
	            System.out.println("Q5");
	            //Get the location position to be moved
	            com.kuka.nav.Position targetPosition = targetLocation.getPose().toPosition();
	            System.out.println("Q6");
	            //Create VirtualLineMotion and go to the pose
	            
	            
	            System.out.println("" + currentLocation);
	            System.out.println("" + targetPosition);
	            VirtualLine v1 = VirtualLine.from(currentLocation).to(targetPosition);
	            System.out.println("Q7");
	            
	            VirtualLineMotion motion = new VirtualLineMotion(kmp.getPose(), targetLocation.getPose());
	            motion.setOrientationMode(OrientationMode.VARIABLE);
	            
	            System.out.println("Q2");
	            	/*
	            VirtualLine Line = null;
	            
	            Line.setGoal(targetLocation.getPose().toPosition());
	            System.out.println("Q3");
	            Line.setStart(kmp.getPose().toPosition());
	            
	            System.out.println("Q4");*/
	            		
	            //Create VirtualLineMotion and go to the pose
	            
	            System.out.println("Executing the motion");

	            //Execute and block until finish
	            try{
	            	kmp.lock();
	            	kmp.execute(motion);
	            }catch (Exception e){
	            	
	            	getLogger().error("Error during motion: ", e);
	                response = "ERROR: Something went wrong during the motion" + e.getMessage(); // Provide more info in the response
	                t.sendResponseHeaders(500, response.length()); // 500 for Internal Server Error
	                OutputStream os = t.getResponseBody();
	                os.write(response.getBytes());
	                os.close();
	            }finally{
	            	kmp.unlock();
	            	System.out.println("Unlocked");
	            }

	            System.out.println("Executed");
	            
	            
	            
	            
	            response = "OK";
				
		        t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
			}
		}
	    
	    
	    class ArrowsMove implements HttpHandler {
	    	@Override
	        public void handle(HttpExchange t) throws IOException {
	    		String response;
	            Map<String, String> params = queryToMap(t.getRequestURI().getQuery());

	            try {
	                double x = Double.parseDouble(params.get("x"));
	                double y = Double.parseDouble(params.get("y"));
	                double Theta = Double.parseDouble(params.get("Theta"));
	                
	                try {
	                	kmp.lock();
	                	kmp.execute(new RelativeMotion(x, y, Theta));
	                }catch(Exception e){
	                	getLogger().info("KMR unlocked.");
	                }finally{
	        			kmp.unlock();
	        			getLogger().info("KMR unlocked.");
	        		}

	                response = "robot moved";
	            
	            } catch (NumberFormatException e) {
	                response = "Invalid input. Please provide 'num1' and 'num2' as numbers in the query parameters.";
	            } catch (NullPointerException e) {
	                response = "Missing parameters. Please provide 'num1' and 'num2' in the query parameters.";
	            }

	    		
	    		
	    		
		    	t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	    	}
	    	
	    }
	    
	    class CountHandler implements HttpHandler {
	        @Override
	        public void handle(HttpExchange t) throws IOException {
	            String response;
	            Map<String, String> params = queryToMap(t.getRequestURI().getQuery());

	            try {
	                double num1 = Double.parseDouble(params.get("num1"));
	                double num2 = Double.parseDouble(params.get("num2"));
	                double sum = num1 + num2;
	                response = String.valueOf(sum);
	            } catch (NumberFormatException e) {
	                response = "Invalid input. Please provide 'num1' and 'num2' as numbers in the query parameters.";
	            } catch (NullPointerException e) {
	                response = "Missing parameters. Please provide 'num1' and 'num2' in the query parameters.";
	            }
	            
	            t.sendResponseHeaders(200, response.length());
	            OutputStream os = t.getResponseBody();
	            os.write(response.getBytes());
	            os.close();
	        }
	    }
	   
	    


		@Override
		public void dispose(){
			server.stop(1);
			run = false;
		}

	    @Override
	    public void run() {
	        System.out.println("Robot Server is running...");
	        
	        gripper.initializeGripper();
			gripper.setSpeed(initSpeed);
			gripper.setForce(initForce);
	        
	       
	        while (run) {
	        	
	        	if (operation == "GotoCartesianposition")
					GotoCartesian(TX, TY, TZ, TA, TB, TC, speed, motion);
				
				else if (operation == "GotoJointposition")
					GotoJoint(A1, A2, A3, A4, A5, A6, A7);
	            // Keep the application running
	        }
	    }
	    



	    // Helper method to parse query parameters from URL
	    private Map<String, String> queryToMap(String query) {
	        Map<String, String> result = new HashMap<String, String>();
	        if (query != null) {
	            for (String param : query.split("&")) {
	                String[] entry = param.split("=");
	                if (entry.length > 1) {
	                    result.put(entry[0], entry[1]);
	                } else {
	                    result.put(entry[0], "");
	                }
	            }
	        }
	        return result;
	    }
	    
	    
	    
	    private void GotoCartesian(double x, double y, double z, double a, double b, double c, double speed, String motion )
		{		
			
			System.out.println("Goto cartesian at pose: " + "x: " + x + ", y: " + y + ", z: " + z + ", a: " + a + ", b: " + ", c: " + c);	
			Frame calculatedFrame = new Frame(x, y, z, a, b, c);
			fail = "";
			
			System.out.println(distanceFrame(calculatedFrame));

			try{
				if ( motion == "ptp" )
				{
					System.out.println("ptp motion");
					robot.move(ptp(calculatedFrame).setJointVelocityRel(speed));
					//vacuumTCP.move(ptp(calculatedFrame).setJointVelocityRel(speed));
		
				}
				else 
				{
					System.out.println("linear motion");
					robot.move(lin(calculatedFrame).setJointVelocityRel(speed));
					//vacuumTCP.move(lin(calculatedFrame).setJointVelocityRel(speed));
		
				}
				
			} 
			catch (CommandInvalidException e){
				System.out.println("Can not reach that point");
				fail = "failed";
			}
					
			
			operation="";

			
		}
		
		private void GotoJoint(double a1, double a2, double a3, double a4, double a5, double a6, double a7)
		{		
			
			System.out.println("Goto Joint method, gping to pos:" + a1 + ", " + a2 + ", " + a3 + ", " + a4 + ", " + a5 + ", " + a6 + ", " + a7);	
			robot.move(ptp(new JointPosition(a1, a2, a3, a4, a5, a6, a7)).setJointVelocityRel(speed));

			operation="";
		}

		public double distanceFrame(Frame desti)
		{
			ObjectFrame actual_pos = robot.getFlange();
			Frame pos = actual_pos.copy();
			double distance = pos.distanceTo(desti);
			return distance;
			
		}
	    
	}