.. _coordinatesystems:

Coordinate Systems
==================

Cameratransform uses three different coordinate systems.

Image
-----
This 2D coordinate system defines a position on the image in pixel. The position X is between 0 and *image_width_px* and
the position y is between 0 and *image_height_px*. (0,0) is the top left corner of the image.

See also :ref:`camprojections`.

.. raw:: html

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .slidecontainer {
            display: inline-block;
        }
        .slidecontainer label {
            display: inline-block;
            width: 125px;
            text-align: right;
            padding-right: 5px;
        }
        .slidecontainer input {
            display: inline-block;
            top: 7px;
            position: relative;
        }
        .slidecontainer span {
            display: inline-block;
            width: 40px;
            text-align: right;
        }
    </style>
    <div style="display: inline-block; width: 300px; position: relative">
       <canvas id="c"></canvas><br>
        <svg style="position:absolute; top: 0; left: 0" viewBox="-1 -1 50 50" width=60>
            <line stroke="#fff" x1=5 x2=5 y1=5 y2=20 />
            <line stroke="#fff" x1=2 x2=5 y1=15 y2=20 />
            <line stroke="#fff" x1=8 x2=5 y1=15 y2=20 />
            <text x=25 y=8 fill="#fff" font-size=11>x</text>

            <line stroke="#fff" x1=5 x2=20 y1=5 y2=5 />
            <line stroke="#fff" x1=15 x2=20 y1=2 y2=5 />
            <line stroke="#fff" x1=15 x2=20 y1=8 y2=5 />
            <text x=2 y=30 fill="#fff" font-size=11>y</text>
        </svg>
       <span id="pos">x=... y=... (hover over image)</span><br>
    </div>
    <div class="slidecontainer">
      <label>image_width_px</label><input type="range" min="100" max="300" value="300" class="slider" id="image_width_px"><span>300</span><br/>
      <label>image_height_px</label><input type="range" min="100" max="300" value="200" class="slider" id="image_height_px"><span>200</span><br/>
      <label>view_x_deg</label><input type="range" min="50" max="180" value="72" class="slider" id="view_x_deg"><span>72</span><br/>
      <br/>
    </div>
    <br/>

Note: in this visualisation *view_x_deg* is used. The field of view can also be specified by *focallength_x_px
= image_width_px / (2 \* tan(view_x_deg / 2))* or *focallength_px = focallength_mm / sensor_width_mm \* image_width_px*.

Space
-----
This 3D coordinate system defines an euclidean space in which the camera is positioned. Distances are in meter.
The camera is positioned at (pos_x_m, pos_y_m, elevation_m) in this coordinate system.
When heading_deg is 0, the camera faces in the positive y direction of this coordinate system.

See also :ref:`CamOrientation`.

.. raw:: html

    <canvas id="c2" style="display: inline-block;"></canvas>
    <div class="slidecontainer">
      <label>pos_x_m</label><input type="range" min="-10" max="10" value="0" step="0.1" class="slider" id="pos_x_m"><span>0</span><br/>
      <label>pos_y_m</label><input type="range" min="-10" max="10" value="-2" step="0.1" class="slider" id="pos_y_m"><span>-1</span><br/>
      <label>elevation_m</label><input type="range" min="0" max="10" value="2" step="0.1" class="slider" id="elevation_m"><span>2</span><br/>
      <br/>
      <label>heading_deg</label><input type="range" min="-180" max="180" value="0" class="slider" id="heading_deg"><span>0</span><br/>
      <label>tilt_deg</label><input type="range" min="00" max="180" value="40" class="slider" id="tilt_deg"><span>40</span><br/>
      <label>roll_deg</label><input type="range" min="-180" max="180" value="0" class="slider" id="roll_deg"><span>0</span><br/>
    </div>
    <script type="module">
        import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/build/three.module.js';
        import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r132/examples/jsm/controls/OrbitControls.js';

        let image_size = [300, 200];
        let cam_params = {image_width_px:300, image_height_px:200, view_x_deg: 72, pos_x_m: 0, pos_y_m: -1, elevation_m: 2, heading_deg: 0, tilt_deg: 40, roll_deg: 0}
        document.getElementById("pos_x_m").oninput = function() {setCamParameter({pos_x_m: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("pos_y_m").oninput = function() {setCamParameter({pos_y_m: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("elevation_m").oninput = function() {setCamParameter({elevation_m: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("heading_deg").oninput = function() {setCamParameter({heading_deg: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("tilt_deg").oninput = function() {setCamParameter({tilt_deg: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("roll_deg").oninput = function() {setCamParameter({roll_deg: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("image_width_px").oninput = function() {setCamParameter({image_width_px: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("image_height_px").oninput = function() {setCamParameter({image_height_px: this.value}); this.nextSibling.innerText = this.value}
        document.getElementById("view_x_deg").oninput = function() {setCamParameter({view_x_deg: this.value}); this.nextSibling.innerText = this.value}
        const event = new Event('cam_update');
        window.cam_params = cam_params;
        window.setCamParameter = function(pos) {
            console.log(pos, cam_params)
            for (var i in pos) {
                            console.log(pos[i], cam_params[i]);
                cam_params[i] = pos[i];
            }
            dispatchEvent(event);
        }


        function createScene(depth) {
            const camera = new THREE.PerspectiveCamera(75, image_size[0]/image_size[1], 0.1, depth);
            camera.position.set(0, 0, 20);
            camera.last_rot = [0, 0, 0];
            addEventListener('cam_update', (e) => {
                if(scene.renderer !== undefined)
                    scene.renderer.setSize( cam_params.image_width_px, cam_params.image_height_px );
                camera.aspect = cam_params.image_width_px/cam_params.image_height_px;
                camera.fov = cam_params.view_x_deg;
                camera.updateProjectionMatrix();
                camera.position.set(cam_params.pos_x_m, cam_params.pos_y_m, cam_params.elevation_m);
                console.log([cam_params.roll_deg, cam_params.tilt_deg, cam_params.heading_deg], camera.last_rot)
                camera.rotateZ(-camera.last_rot[0]*Math.PI/180);
                camera.rotateX(-camera.last_rot[1]*Math.PI/180);
                camera.rotateZ(-camera.last_rot[2]*Math.PI/180);

                camera.rotateZ(-cam_params.heading_deg*Math.PI/180);
                camera.rotateX(cam_params.tilt_deg*Math.PI/180);
                camera.rotateZ(cam_params.roll_deg*Math.PI/180);
                camera.last_rot = [cam_params.roll_deg, cam_params.tilt_deg, -cam_params.heading_deg];
                scene.render()
            }, false);


            const scene = new THREE.Scene();
            const geometry = new THREE.BoxGeometry(1, 1, 1);
            const material = new THREE.MeshPhongMaterial({color: 0x44aa88});  // greenish blue
            const cube = new THREE.Mesh(geometry, material);
            cube.position.set(0, 0, 0.5)
            scene.add(cube);

            const material_line = new THREE.LineBasicMaterial( { color: 0x404040 } );
            scene.material_line = material_line;
            const points = [];
            for(let x = -10; x<=10 ; x+=2) {
                points.push( new THREE.Vector3(x,   -10, 0 ) );
                points.push( new THREE.Vector3(x,    10, 0 ) );
                if(x<10) {
                    points.push(new THREE.Vector3(x + 1, 10, 0));
                    points.push(new THREE.Vector3(x + 1, -10, 0));
                }
            }
            const geometry_line = new THREE.BufferGeometry().setFromPoints( points );
            const line = new THREE.Line( geometry_line, material_line );
            scene.add( line );
            const points2 = [];
            for(let y = -10; y<=10 ; y+=2) {
                points2.push( new THREE.Vector3(-10, y, 0 ) );
                points2.push( new THREE.Vector3( 10, y, 0 ) );
                if(y< 10) {
                    points2.push( new THREE.Vector3( 10, y+1, 0 ) );
                    points2.push( new THREE.Vector3(-10, y+1, 0 ) );
                }
            }

            const geometry_line2 = new THREE.BufferGeometry().setFromPoints( points2);
            const line2 = new THREE.Line( geometry_line2, material_line );
            scene.add( line2 );

            {
                const color = 0xFFFFFF;
                const intensity = 1;
                const light = new THREE.DirectionalLight(color, intensity);
                light.position.set(1, -2, 4);
                scene.add(light);
              }

            return [camera, scene];
        }

        function main2(id, font) {
            const canvas = document.querySelector(id);
            const renderer = new THREE.WebGLRenderer({canvas});

            const [camera, scene] = createScene(1);
            camera.near = 0.9;
            scene.add(camera)

            const camera_extern = new THREE.PerspectiveCamera(75, 1, 0.1, 5000);
            renderer.setSize( 300, 300 );

            //camera_extern.rotateX(45*Math.PI/180);
            //camera_extern.rotateZ(90*Math.PI/180);

            const controls = new OrbitControls(camera_extern, renderer.domElement);
            //controls.maxPolarAngle = Math.PI * 0.5;
            //controls.minDistance = 10;
            //controls.maxDistance = 50;
            console.log("Hi");


            const color = 0xA0A6A9;
            const matDark = new THREE.LineBasicMaterial( {
                    color: color,
                    side: THREE.DoubleSide
                } );
            const shapes_y = font.generateShapes( "y", 0.5 );
            const geometry_y = new THREE.ShapeGeometry( shapes_y );
            const text_y = new THREE.Mesh( geometry_y, matDark );
            text_y.position.y = 2.5;
            scene.add( text_y);
            const shapes_x = font.generateShapes( "x", 0.5 );
            const geometry_x = new THREE.ShapeGeometry( shapes_x );
            const text_x = new THREE.Mesh( geometry_x, matDark );
            text_x.position.x = 2.5;
            scene.add( text_x);
            const shapes_z = font.generateShapes( "z", 0.5 );
            const geometry_z = new THREE.ShapeGeometry( shapes_z );
            const text_z = new THREE.Mesh( geometry_z, matDark );
            text_z.position.z = 2.5;
            text_z.rotateX(90*Math.PI/180);
            text_z.rotateY(90*Math.PI/180);
            scene.add( text_z);

            var points2 = [];
            points2.push( new THREE.Vector3(0, 0, 0 ) );
            points2.push( new THREE.Vector3(0, 0, 10 ) );
            var geometry_line2 = new THREE.BufferGeometry().setFromPoints( points2);
            var line2 = new THREE.Line( geometry_line2, scene.material_line );
            scene.add( line2 );

            const material_line2 = new THREE.LineBasicMaterial( { color: 0xF0F0F0 } );
            points2 = [];
            const w = 0.3;
            points2.push( new THREE.Vector3(0.1, 0, 0 ) );
            points2.push( new THREE.Vector3(0.1, 0, 2 ) );
            points2.push( new THREE.Vector3(0.1, 0-w/2, 2-w) );
            points2.push( new THREE.Vector3(0.1, 0, 2 ) );
            points2.push( new THREE.Vector3(0.1, 0+w/2, 2-w) );
            geometry_line2 = new THREE.BufferGeometry().setFromPoints( points2);
            line2 = new THREE.Line( geometry_line2, material_line2 );
            scene.add( line2 );

            points2 = [];
            points2.push( new THREE.Vector3(0.1, 0, 0 ) );
            points2.push( new THREE.Vector3(0.1, 2, 0 ) );
            points2.push( new THREE.Vector3(0.1-w/2, 2-w, 0) );
            points2.push( new THREE.Vector3(0.1, 2, 0 ) );
            points2.push( new THREE.Vector3(0.1+w/2, 2-w, 0) );
            geometry_line2 = new THREE.BufferGeometry().setFromPoints( points2);
            line2 = new THREE.Line( geometry_line2, material_line2 );
            scene.add( line2 );

            points2 = [];
            points2.push( new THREE.Vector3(0.1, 0, 0 ) );
            points2.push( new THREE.Vector3(2.1, 0, 0 ) );
            points2.push( new THREE.Vector3(2.1-w, 0-w/2, 0) );
            points2.push( new THREE.Vector3(2.1, 0, 0 ) );
            points2.push( new THREE.Vector3(2.1-w, 0+w/2, 0) );
            geometry_line2 = new THREE.BufferGeometry().setFromPoints( points2);
            line2 = new THREE.Line( geometry_line2, material_line2 );
            scene.add( line2 );


            camera_extern.position.set(6, 0, 3);
            //camera_extern.rotateX(45*Math.PI/180);
            camera_extern.rotateZ(90*Math.PI/180);
            camera_extern.rotateX(80*Math.PI/180);

            camera.updateProjectionMatrix()
            const cameraPerspectiveHelper = new THREE.CameraHelper(camera);
            scene.add(cameraPerspectiveHelper);

            camera.updateProjectionMatrix()
            cameraPerspectiveHelper.update()

            cameraPerspectiveHelper.visible = true;
            renderer.render(scene, camera_extern);
            scene.render = () =>         renderer.render(scene, camera_extern);


            function animate(now) {
                requestAnimationFrame(animate);
                cameraPerspectiveHelper.update();
                renderer.render(scene, camera_extern);
            }

            animate(0);
        }

        function main(id) {
            const canvas = document.querySelector(id);
            const renderer = new THREE.WebGLRenderer({canvas});
            renderer.setSize( image_size[0], image_size[1] );

            const [camera, scene] = createScene(100);

            var ctx = canvas.getContext("2d");
            scene.render = () => {
                renderer.render(scene, camera);
            }
            scene.render();
            scene.renderer = renderer;
        }


        const loader = new THREE.FontLoader();
        var font = undefined;
        loader.load( 'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function ( f ) {
            font = f;
            console.log("font", font);

            main("#c");
            main2("#c2", font);
            setCamParameter(cam_params);
        } );

        let scale = 1;
        var canvas = document.getElementById("c");
        canvas.onmousemove = function (e) {
            let rect = e.target.getBoundingClientRect();
            let x = e.clientX - rect.left; //x position within the element.
            let y = e.clientY - rect.top;  //y position within the element.
            document.getElementById("pos").innerText = `x=${(x / scale).toFixed(1)} y=${(y / scale).toFixed(1)}`;
        }

    </script>

GPS
---
This is a geo-coordinate system in which the camera is positioned. The coordinates are latitude, longitude and elevation.
The camera is positioned at (lat, lon). When heading_deg is 0, the camera faces north in this coordinate system.

This coordinate system shares the parameters elevation_m and the orientation angles (heading_deg, tilt_deg, roll_deg)
with the **Space** coordinate system to keep both coordinate systems aligned.

See also :ref:`gps`.

