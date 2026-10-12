import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import './App.css';

const initialData = {
  step: 0,
  totalSteps: 100,
  jointStates: [],
  eePosition: [0, 0, 0],
  ultrasound: null,
  aiMask: null,
  hemorrhageDetected: false,
  confidence: 0,
  history: [],
};

function RobotArm({ jointStates }) {
  const jointAngle = (index) => jointStates[index]?.position ?? 0;

  return (
    <group position={[0, 0, 0]}>
      <mesh position={[0, 0.035, 0]}>
        <cylinderGeometry args={[0.09, 0.1, 0.07, 32]} />
        <meshStandardMaterial color="#414752" metalness={0.55} roughness={0.35} />
      </mesh>
      <group position={[0, 0.07, 0]} rotation={[0, 0, jointAngle(0)]}>
        <mesh position={[0, 0.13, 0]}>
          <boxGeometry args={[0.055, 0.26, 0.055]} />
          <meshStandardMaterial color="#ff6b24" metalness={0.35} roughness={0.4} />
        </mesh>
        <mesh position={[0, 0.27, 0]}>
          <sphereGeometry args={[0.045, 20, 20]} />
          <meshStandardMaterial color="#c9d0d8" metalness={0.65} roughness={0.3} />
        </mesh>
        <group position={[0, 0.27, 0]} rotation={[0, 0, jointAngle(1)]}>
          <mesh position={[0, 0.115, 0]}>
            <boxGeometry args={[0.045, 0.23, 0.045]} />
            <meshStandardMaterial color="#ffad33" metalness={0.35} roughness={0.4} />
          </mesh>
          <mesh position={[0, 0.24, 0]}>
            <sphereGeometry args={[0.038, 20, 20]} />
            <meshStandardMaterial color="#c9d0d8" metalness={0.65} roughness={0.3} />
          </mesh>
          <group position={[0, 0.24, 0]} rotation={[0, 0, jointAngle(2)]}>
            <mesh position={[0, 0.08, 0]}>
              <cylinderGeometry args={[0.017, 0.025, 0.16, 16]} />
              <meshStandardMaterial color="#00d4ff" metalness={0.35} roughness={0.3} />
            </mesh>
          </group>
        </group>
      </group>
    </group>
  );
}

function TissuePhantom() {
  return (
    <group position={[0.37, 0, 0]}>
      <mesh position={[0, 0, -0.015]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.3, 0.3]} />
        <meshStandardMaterial color="#b78c72" roughness={0.8} />
      </mesh>
      <mesh position={[0.03, 0.005, 0]}>
        <sphereGeometry args={[0.015, 16, 16]} />
        <meshStandardMaterial color="#a83232" transparent opacity={0.75} />
      </mesh>
    </group>
  );
}

function App() {
  const [connected, setConnected] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const [data, setData] = useState(initialData);

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let isActive = true;

    const connect = () => {
      if (!isActive) return;

      socket = new WebSocket('wss://rahat-project.onrender.com/ws');

      socket.onopen = () => {
        setConnected(true);
        setConnectionError('');
      };

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          setData((previous) => ({
            step: msg.step,
            totalSteps: msg.total_steps,
            jointStates: msg.joint_states,
            eePosition: msg.end_effector_position,
            ultrasound: `data:image/jpeg;base64,${msg.ultrasound_frame}`,
            aiMask: `data:image/png;base64,${msg.ai_mask}`,
            hemorrhageDetected: msg.hemorrhage_detected,
            confidence: msg.confidence,
            history: [
              ...previous.history,
              {
                step: msg.step,
                confidence: msg.confidence,
                detected: msg.hemorrhage_detected ? 1 : 0,
              },
            ].slice(-100),
          }));
        } catch {
          setConnectionError('Received an invalid message from the backend.');
        }
      };

      socket.onerror = () => {
        setConnectionError('Unable to connect. Check that the backend is running on port 8000.');
      };

      socket.onclose = () => {
        setConnected(false);
        if (isActive) reconnectTimer = window.setTimeout(connect, 1500);
      };
    };

    connect();

    return () => {
      isActive = false;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return (
    <main className="dashboard">
      <header className="header">
        <div>
          <p className="eyebrow">ROBOTICS · SIMULATION · COMPUTER VISION</p>
          <h1>RAHAT <span>Robotic Ultrasound Simulation</span></h1>
        </div>
        <div className={`status ${connected ? 'connected' : 'disconnected'}`} role="status">
          <span className="status-dot" />
          {connected ? 'Live' : 'Disconnected'}
        </div>
      </header>

      {connectionError && <p className="connection-error">{connectionError}</p>}

      <div className="demo-notice">
        Research prototype only. The model uses random, untrained weights; predictions are not
        medically meaningful.
      </div>

      <div className="main-grid">
        <section className="panel robot-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">LIVE SIMULATION</p>
              <h2>Robotic Scanner</h2>
            </div>
            <span className="panel-index">01</span>
          </div>
          <div className="canvas-wrap">
            <Canvas camera={{ position: [0.9, 0.75, 1.15], fov: 42 }}>
              <color attach="background" args={['#0d1118']} />
              <ambientLight intensity={0.8} />
              <pointLight position={[1, 1.5, 1]} intensity={35} />
              <RobotArm jointStates={data.jointStates} />
              <TissuePhantom />
              <gridHelper args={[1.2, 12, '#27313e', '#1b232e']} position={[0, -0.003, 0]} />
              <OrbitControls enablePan enableZoom />
            </Canvas>
          </div>
          <div className="info-row">
            <span>SCAN STEP <strong>{data.step} / {data.totalSteps}</strong></span>
            <span>END EFFECTOR <strong>({data.eePosition.map((value) => value.toFixed(3)).join(', ')})</strong></span>
          </div>
        </section>

        <section className="panel imaging-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">INFERENCE OUTPUT</p>
              <h2>Ultrasound + AI Mask</h2>
            </div>
            <span className="panel-index">02</span>
          </div>
          <div className="image-container">
            {data.ultrasound ? (
              <>
                <img src={data.ultrasound} alt="Simulated ultrasound frame" className="ultrasound-img" />
                {data.aiMask && <img src={data.aiMask} alt="Untrained model output mask" className="mask-img" />}
              </>
            ) : (
              <div className="image-placeholder">Waiting for simulation frames…</div>
            )}
          </div>
          <div className={`detection-badge ${data.hemorrhageDetected ? 'alert' : 'safe'}`}>
            <span>{data.hemorrhageDetected ? '!' : '✓'}</span>
            {data.hemorrhageDetected ? 'Model threshold exceeded' : 'Model threshold not exceeded'}
            <strong>{(data.confidence * 100).toFixed(1)}%</strong>
          </div>
        </section>

        <section className="panel chart-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">SCAN HISTORY</p>
              <h2>Model Confidence</h2>
            </div>
            <span className="panel-index">03</span>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.history} margin={{ top: 10, right: 16, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="#252c37" strokeDasharray="3 3" />
                <XAxis dataKey="step" stroke="#77808d" tickLine={false} axisLine={false} />
                <YAxis domain={[0, 1]} stroke="#77808d" tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ background: '#171c25', border: '1px solid #303947', borderRadius: 8 }}
                  labelStyle={{ color: '#cbd3dd' }}
                />
                <Line type="monotone" dataKey="confidence" stroke="#00d4ff" strokeWidth={2} dot={false} />
                <Line type="step" dataKey="detected" stroke="#ff5b63" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-legend">
            <span><i className="legend-confidence" />Confidence</span>
            <span><i className="legend-threshold" />Threshold status</span>
          </div>
        </section>
      </div>
    </main>
  );
}

export default App;
