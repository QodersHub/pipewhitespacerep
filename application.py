from flask import Flask, render_template, request, session, abort, flash, redirect, url_for, json, jsonify
from flask_wtf import FlaskForm, Form
from os import listdir
from os.path import isfile, join
from wtforms import StringField, PasswordField, IntegerField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import progressbar
from os import listdir
from os.path import isfile, join
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tqdm import tqdm_notebook
from scipy import misc
from scipy.ndimage import rotate
import copy
import matplotlib.image as mpimg
import application
from numpy.linalg import norm
import sys
import copy
from well_planner.scenarios import *
from well_planner import Planner
import well_planner
from numpy.linalg import norm
import sys
import os
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm, trange
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from os.path import isfile, join
from skfuzzy import control as ctrl
import csv
import utm

# Folder path and file extesion
UPLOAD_FOLDER1 = './input/TIFF/File1'
UPLOAD_FOLDER2 = './input/TIFF/File2'
ALLOWED_EXTENSIONS = set(['png'])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisisasecret!'
# need to know what app.confif do
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2

# Configure Uploaded Folder
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER -- Not working!
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)
# To check existing file within a folder
existing_files1 = [f for f in listdir(UPLOAD_FOLDER1) if isfile(join(UPLOAD_FOLDER1, f))]
existing_files2 = [f for f in listdir(UPLOAD_FOLDER2) if isfile(join(UPLOAD_FOLDER2, f))]

# Upload File Section
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# To check file uploaded has the same name or already uploaded or not 
def existing_file1(filename):
    if filename in existing_files1:
        return True
    else:
        return False

def existing_file2(filename):
    if filename in existing_files2:
        return True
    else:
        return False

@app.route('/pipeline', methods=['POST','GET'])
def pipeline():
    form = ParameterForm()
    target1 = os.path.join(APP_ROOT,'input/TIFF/File1/')
    target2 = os.path.join(APP_ROOT,'input/TIFF/File2/')
    
    if request.method == 'POST':
        # check if the target file exist 
        if not os.path.isdir(target1):
                os.mkdir(target1)

        if not os.path.isdir(target2):
                os.mkdir(target2)
            # Testing purposes to check how many file got in
            # csv is coming from the form 'from' pipeline page
        newfiles1 = request.files.getlist("png1") #png1 is coming from front end form input file field called png1
        newfiles2 = request.files.getlist("png2") #png2 is coming from front end form input file field called png2
       
        print(f'{len(newfiles1)} files found.:{newfiles1}') # Check correct file being uploaded
        print(f'{len(newfiles2)} files found.:{newfiles2}') # Check correct file being uploaded
    
        # File 1 Check Validity function
        for file in newfiles1:
            print(f'{(file)} files found.')
            if allowed_file(file.filename):
                print(f'File/s allowed and saved to file folder')
                if existing_file1(file.filename):
                    print(f'... skipping file copy - file already exist.')
                    geohazimage = file.filename
                else:
                    filename = secure_filename(file.filename)
                    print(f'... copying file')
                    file.save(os.path.join(app.config['UPLOAD_FOLDER1'], filename))
                    geohazimage = file.filename
            
            else:
                print(f'Skipping {file.filename} - filetype not allowed (not a csv file).')
           
        # File 2 Check Validity function
        for file in newfiles2:
            print(f'{(file)} files found.')
            if allowed_file(file.filename):
                print(f'File/s allowed and saved to file folder')
                if existing_file2(file.filename):
                    print(f'... skipping file copy - file already exist.')
                    pipelineimage = file.filename
                else:
                    filename = secure_filename(file.filename)
                    print(f'... copying file')
                    file.save(os.path.join(app.config['UPLOAD_FOLDER2'], filename))
                    pipelineimage = file.filename
            
            else:
                print(f'Skipping {file.filename} - filetype not allowed (not a csv file).')
            
        print("Pass Through File checking before Machine Learning")
        return redirect(machineLearning(pgeohazimage = geohazimage,ppipelineimage = pipelineimage,pgeohaz_xori=form['geohaz_xori'].data,pgeohaz_yori=form['geohaz_yori'].data,ppipelines_xori=form['pipelines_xori'].data,ppipelines_yori=form['pipelines_yori'].data,px_strt=form['x_strt'].data,py_strt=form['y_strt'].data,px_end=form['x_end'].data,py_end=form['y_end'].data))
    else:
        print('here')  
        return render_template('pipeline.html', form=form)
    

# User Login Section
class LoginForm(FlaskForm):
    username = StringField('username',validators=[InputRequired(message='A username is required')])
    password = PasswordField('password',validators=[InputRequired(message='Password is required')])

class ParameterForm(FlaskForm):
    geohaz_xori = IntegerField('geohaz_xori',validators=[InputRequired(message='An Integer is required')])
    geohaz_yori = IntegerField('geohaz_yori',validators=[InputRequired(message='An Integer is required')])
    pipelines_xori = IntegerField('pipelines_xori',validators=[InputRequired(message='An Integer is required')])
    pipelines_yori = IntegerField('pipelines_yori',validators=[InputRequired(message='An Integer is required')])
    x_strt = IntegerField('x_strt',validators=[InputRequired(message='An Integer is required')])
    y_strt = IntegerField('y_strt',validators=[InputRequired(message='An Integer is required')])
    x_end = IntegerField('x_end',validators=[InputRequired(message='An Integer is required')])
    y_end = IntegerField('y_end',validators=[InputRequired(message='An Integer is required')])

rev='6'
csvtarget = os.path.join(APP_ROOT,f'output/trajectories/rev{rev}_mrg/rev{rev}_fulltrees.csv')

@app.route('/machineLearning')
def machineLearning(pgeohazimage,ppipelineimage,pgeohaz_xori,pgeohaz_yori,ppipelines_xori,ppipelines_yori,px_strt,py_strt,px_end,py_end): 
    rev='6'
    
    ###############_____________IMPORT GEOHAZARD GRID_________________################
    #Import the Seabed Obstructions (Geohazard) image
    geohaz_im = mpimg.imread(f'input/TIFF/File1/{pgeohazimage}')
    #print(f'Image loaded with shape {geohaz_im.shape} - total number of samples {geohaz_im.shape[0]*geohaz_im.shape[1]}')

    geohaz_im = (geohaz_im)/(geohaz_im.max())
    # np.save(f'output/trajectories/rev{rev}/test.csv',geohaz_im)
    # geohaz_im.to_csv(f'output/trajectories/rev{rev}/test.csv')
    
    #Create XY placeholders for the geohazard
    geohaz_x=np.zeros_like(geohaz_im)
    geohaz_y=np.zeros_like(geohaz_im)

    #Populate placeholders
    # From TSM:
    geohaz_xori=pgeohaz_xori
    geohaz_yori=pgeohaz_yori
    geohaz_dx=10
    geohaz_dy=-10

    # Populate XYs
    for col in range(geohaz_x.shape[1]):
        geohaz_x[:,col]=col*geohaz_dx+geohaz_xori

    for row in range(geohaz_y.shape[0]):
        geohaz_y[row,:]=row*geohaz_dy+geohaz_yori

    # Populate geohazards DF
    geohaz_xarray =geohaz_x.reshape(-1,1)
    geohaz_yarray =geohaz_y.reshape(-1,1)
    geohaz_imarray=geohaz_im.reshape(-1,1)

    geohaz_df=pd.DataFrame(np.column_stack((geohaz_xarray,geohaz_yarray,geohaz_imarray)))
    geohaz_df.columns=['X','Y','IM']

    #Scale Geohazards DF
    geohaz_df['IM']=geohaz_df['IM']/max(geohaz_df['IM'])
    # testing purposes
    # return url_for('contactus')
    ###############_____________IMPORT PIPELINES GRID_________________################

    ##Import the Pipelines tiff
    pipelines_im = mpimg.imread(f'input/TIFF/File2/{ppipelineimage}')
    #print(f'Image loaded with shape {pipelines_im.shape} - total number of samples {pipelines_im.shape[0]*pipelines_im.shape[1]}')

    #Create XY placeholders for the pipelines
    pipelines_x=np.zeros_like(pipelines_im)
    pipelines_y=np.zeros_like(pipelines_im)

    #Populate Placeholders 
    # From TSM:
    pipelines_xori=ppipelines_xori
    pipelines_yori=ppipelines_yori
    pipelines_dx=10
    pipelines_dy=-10

    #
    # Populate XYs
    for col in range(pipelines_x.shape[1]):
        pipelines_x[:,col]=col*pipelines_dx+pipelines_xori

    for row in range(pipelines_y.shape[0]):
        pipelines_y[row,:]=row*pipelines_dy+pipelines_yori

    #Populate pipeline DF
    pipelines_xarray =pipelines_x.reshape(-1,1)
    pipelines_yarray =pipelines_y.reshape(-1,1)
    pipelines_imarray=pipelines_im.reshape(-1,1)

    pipelines_df=pd.DataFrame(np.column_stack((pipelines_xarray,pipelines_yarray,pipelines_imarray)))
    pipelines_df.columns=['X','Y','IM']

    #Scale Pipeline DF
    pipelines_df['IM']=pipelines_df['IM']/max(pipelines_df['IM'])

    #Set the Start and Stop Coordinates --- maybe interactive
    x_strt,y_strt = px_strt,py_strt
    x_end,y_end   = px_end,py_end

    #######__________CREATE REWARD GRID___________##########

    #Set sampling for the reward grid at every 20m
    dx=20
    dy=20

    #Initialize reward grid
    # Retrieve extent of final grid
    xmin = geohaz_df.X.min()
    xmax = geohaz_df.X.max()
    ymin = geohaz_df.Y.min()
    ymax = geohaz_df.Y.max()
    #
    nx=int((xmax-xmin)/float(dx))+1
    ny=int((ymax-ymin)/float(dy))+1
    rewardgrid=np.zeros((nx,ny))
    #
    #print('Final grid ')
    #print('... has    '+str([nx,ny])+' samples in x,y,z')
    #print('... covers '+str([int(xmax-xmin),int(ymax-ymin)])+' metres in x,y')
    #print('... has    '+str(nx*ny)+' samples')


    #Populate grid with geohazards
    buff_x,buff_y=0,0 # anti-collision distance expressed in cells
    geohaz_df_nogo=geohaz_df[geohaz_df['IM']==0].copy()

    i=0
    cnt=0
    with progressbar.ProgressBar(max_value=len(geohaz_df_nogo)) as bar:
        
        for row in range(len(geohaz_df_nogo)):
            #
            Xidx=((geohaz_df_nogo.iloc[row]['X']-xmin)/float(dx)).astype(int)
            Yidx=((geohaz_df_nogo.iloc[row]['Y']-ymin)/float(dy)).astype(int)
            #
            rewardgrid[Xidx-buff_x:Xidx+buff_x+1,Yidx-buff_y:Yidx+buff_y+1]=-np.inf
            cnt+=1
            
            # bar update
            bar.update(i)
            i+=1

    
    #Rotate the reward grid
    # get start/stop indices
    x_strt_idx=((x_strt-xmin)/float(dx)).astype(int)
    y_strt_idx=((x_end-xmin) /float(dx)).astype(int)
    x_end_idx =((y_strt-ymin)/float(dy)).astype(int)
    y_end_idx =((y_end-ymin) /float(dy)).astype(int)

    # get rotation angle
    angle=90-(np.arctan((y_end_idx-y_strt_idx)/(x_end_idx-x_strt_idx)))/np.pi*180.

    # Rotate map
    rewardgrid_tmp=copy.deepcopy(rewardgrid)
    rewardgrid_tmp[np.isinf(rewardgrid_tmp)]=1
    rewardgrid_rot=rotate(rewardgrid_tmp,angle)
    rewardgrid_rot[rewardgrid_rot>0.5]=-np.inf
    rewardgrid_rot[~np.isinf(rewardgrid_rot)]=0.

    # Rotate start point
    image_temp=np.zeros_like(rewardgrid)
    image_temp[x_strt_idx,y_strt_idx]=1
    image_temp=rotate(image_temp,angle)
    x_strt_idx_rot,y_strt_idx_rot=np.argwhere(image_temp>0.5).squeeze()

    # Rotate end point
    image_temp=np.zeros_like(rewardgrid)
    image_temp[x_end_idx,y_end_idx]=1
    image_temp=rotate(image_temp,angle)
    x_end_idx_rot,y_end_idx_rot=np.argwhere(image_temp>0.5).squeeze()

    ##Define Rotation function for vectors for later retrieval of trajectories
    def rotate_array(grid,grid_rot,vec,angle):
        
        xy=[vec[0],vec[1]]
        org_center = (np.array(grid.T.shape[:2][::-1])-1)/2.
        rot_center = (np.array(grid_rot.T.shape[:2][::-1])-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) - org[1]*np.sin(a), org[0]*np.sin(a) + org[1]*np.cos(a) ])
        return new+rot_center

    np.save(f'output/rewardgrids/rewards_rev{rev}.npy', rewardgrid_rot)
    

    ##########_______________PART DEUX_______________#############
    #
    # RRT*
    # Pipeline Planner
    # rev x5:
    #   20m sampling
    #   rotated grid
    #
    #   wider target box

    # This forces Python to import the package from the parent directory
    # If you want to use the pip installed version, remove this line
    #sys.path.insert(0, '..')

    # revision
    rev='6'

    # hyperparameters
    grdscl=10
    maxang=50.
    expdis=200
    goalsk=70
    fwdsam=10.

    print(f'Retrieved well_planner package from {os.path.abspath(well_planner.__file__)}')
    
    # set grids - rewards
    grid = rewardgrid_rot
    print(f'Reward grid size: {grid.shape}')

    # reshape
    from numpy import newaxis
    grid=grid[:,newaxis,:]
    print(f'Reward grid size: {grid.shape}')

    # set start/stop
    strt_pos=[2733, 0,  394]
    end_pos =[2733, 0, 4605]
    print(f'Planning from {strt_pos} ...')
    print(f'       ... to {end_pos}')

    if strt_pos[2]>end_pos[2]: # flip strt/end
        print('Flipping start/end...')
        tmp=strt_pos
        strt_pos=end_pos
        end_pos=tmp
        print('Correction:')
        print(f'Planning from {strt_pos} ...')
        print(f'       ... to {end_pos}')

    # set start direction
    initial_dir=np.array([end_pos[0]-strt_pos[0],end_pos[1]-strt_pos[1],end_pos[2]-strt_pos[2]])
    print(f'Initial direction: {initial_dir}')

    # Parameter print
    print(f'Running with parameters {grdscl}, {maxang}, {expdis}, {goalsk}.')

    # set reward
    xbuff=150
    ybuff=1
    # grid[strt_pos[0]-buff:strt_pos[0]+buff+1, strt_pos[1], strt_pos[2]-buff:strt_pos[2]+buff+1] = -1
    #grid[end_pos[0]-xbuff:end_pos[0]+xbuff+1,   end_pos[1],  end_pos[2]-ybuff:end_pos[2]+ybuff+1 ]  = grdscl*(grid.shape[0]+grid.shape[1]+grid.shape[2])
    grid[end_pos[0]-int(expdis/2):end_pos[0]+int(expdis/2)+1,   end_pos[1],  end_pos[2]:end_pos[2]+expdis]  = grdscl*(grid.shape[0]+grid.shape[1]+grid.shape[2])
    #grid[:,end_pos[1],end_pos[2]+10:]=-np.inf
    #grid = grid[:,:,:end_pos[2]+1]

    print(f'Min/Max in grid found: {grid.min()}/{grid.max()}')

    # grid adjustments
    grid = -grid            # cost instead of reward
    grid[grid==0]=1         # set background travel cost

    # subselect goals from maps
    print(f'Selecting all goals')
    goals = np.argwhere(grid<0)
    print(f'{len(goals)} goals found at {goals}')

    # RRT* params
    expand_dis = expdis
    show_animation = False

    # planner
    print('Setting up the planner...')
    rrt = Planner(grid,
                goals=goals,
                seed=200,
                angle_treshold = (np.pi/30)/maxang,  # 6deg per maxang cells
                neighbourhood = 2.4*expand_dis,      #
                expand_dis=expand_dis,               # dictates linear sampling roughness
                forward_sample_rate = fwdsam,
                length_penalty_coeff = 1,
                goal_sample_rate=goalsk)             # explore/exploit


    #surface_locs=[[strt_pos[0],strt_pos[1],strt_pos[2]]]
    surface_locs=np.array([strt_pos[0],strt_pos[1],strt_pos[2]])
    print("Let's go...")
    #nodes = rrt.plan(surface_locs, 2, num_occ=[1], animate_while_compute=False, animate_at_end=True, cost_to_file=None)
    lowest_cost_nodes=[]
    nodes = rrt.plan_single_well(surface_locs, 1000, direct=initial_dir, animate_while_compute=False, animate_at_end=False, cost_to_file=None,lowest_cost_nodes=lowest_cost_nodes)

    viable_paths = [node for node in nodes if node.cost < 0]

    # output result
    if True:
        print(f'Done! Writing output...')
        for i, path in enumerate(lowest_cost_nodes):
            #for i, path in enumerate(viable_paths):
            path_ip=path.get_full_path_to_root()
            #path_vert=path.get_full_path_to_root(interpolate=False)
            np.savetxt(f'output/trajectories/rev{rev}/path{i}_cost_{round(path.cost)}_length_{round(path.get_length_to_root())}_ip.txt',path_ip)
    #np.savetxt(f'trajs/rev{rev}/path{i}_grdscl_{grdscl}_maxang_{maxang}_expdis_{expdis}_goalsk_{goalsk}_cost_{round(path.cost)}_length_{round(path.get_length_to_root())}_vert.txt',path_vert)

    ####______Merge the fulltrees_______####
    rev = '6'

    inputdir=f'output/trajectories/rev{rev}/'

    #
    print(f'Starting merge ...')
    cnt=0
    isFirst=True
    trees = [f for f in listdir(inputdir) if ('_ip' in f)]
    print(f'{len(trees)} files found. Combining...')
    with tqdm(total = len(trees), unit = 'nodes') as pbar:
        for tree in trees:
            try:
                #print(tree)
                thispath=pd.DataFrame(np.loadtxt(inputdir+tree),columns=['X','Y','Z'])
                thispath['cost']=tree.split('_')[2]
                thispath['len']=tree.split('_')[4]
                thispath['trajcnt']=cnt
                cnt+=1
                
                if isFirst:
                    isFirst=False
                    df=thispath.copy()
                else:
                    df=pd.concat([df,thispath]).copy()
                del thispath
            except:
                None
            
            pbar.update(1)

    print(f'Successfully combined {cnt} trajectories.')
    print(f'Total df length is {len(df)}.')
    df.to_csv(f'output/trajectories/rev{rev}_mrg/rev{rev}_fulltrees.csv')

    ##########_______________PART TRES_______________#############
    rev = '6'

    #Load output from Part 2
    #Traj Directory
    traj_dir_fulltree=f'output/trajectories/rev{rev}_mrg/'

    #Traj Files
    traj_files_fulltree = [m for m in listdir(traj_dir_fulltree) if isfile(join(traj_dir_fulltree, m)) and 'fulltrees' in m]

    #Rotate back to real world XY
    def rotate_back_df_x(df):
        v=np.array(rotate_array(rewardgrid_rot,rewardgrid,[df['Xidx_rot'],df['Yidx_rot']],-angle))
        return v[0]
    def rotate_back_df_y(df):
        v=np.array(rotate_array(rewardgrid_rot,rewardgrid,[df['Xidx_rot'],df['Yidx_rot']],-angle))
        return v[1]

    ###__________Cluster Tree____________###
    tqdm_notebook().pandas()
    isFirst=True
    run_label=0
    for traj_file in traj_files_fulltree:
        #
        # load traj
        this_tree=pd.read_csv(traj_dir_fulltree+traj_file)
        
        #
        this_tree['runID']=run_label
        run_label+=1
        #
        # set dtype
        for col in this_tree.columns:
            this_tree[col]=pd.to_numeric(this_tree[col],errors='coerce')
        #
        this_tree['Xidx_rot']=this_tree['X'].astype(int)
        this_tree['Yidx_rot']=this_tree['Z'].astype(int)
        
        #
        # rotate back to world XY
        this_tree['Xidx'] = this_tree.progress_apply(rotate_back_df_x,axis=1)
        this_tree['Yidx'] = this_tree.progress_apply(rotate_back_df_y,axis=1)

        #
        this_tree['X']=this_tree['Xidx']*float(dx)+xmin
        this_tree['Y']=this_tree['Yidx']*float(dy)+ymin
        
        # merge
        if isFirst:
            isFirst=False
            df_trees=this_tree.copy()
        else:
            df_trees=pd.concat([df_trees,this_tree]).copy()
        del this_tree
    pipelines_df['Xidx']=((pipelines_df['X']-xmin)/float(dx)).astype(int)
    pipelines_df['Yidx']=((pipelines_df['Y']-xmin)/float(dx)).astype(int)


    df_trees.to_csv(traj_dir_fulltree+'alltrees.csv')

    #Changing the Easting Northings into Latitude Longitude
    def getUTMs(row):
        tup = utm.to_latlon(row.ix[0],row.ix[1],50, "N")
        return pd.Series(tup[:2])

    df_trees[['utmminy_lat','utmminx_lng']] = df_trees[['Y','X']].apply(getUTMs , axis=1)

    df_trees.to_csv(traj_dir_fulltree+'alltrees_LAT_LNG.csv')
    return url_for('visualisation')

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    form = LoginForm()
    # Testing on passing data from loginform and it runs the get method first 
    userlogin = "admin"
    passwordlogin = "password"
    x=form['password'].data
    if form.validate_on_submit() and request.method == 'POST':
        # return '<h1>The username is {}. The password is {}.'.format(form.username.data,form.password.data)

        print(userlogin)
        if form['username'].data == userlogin and form['password'].data == passwordlogin:
            return render_template('Modules.html',form=form)
        else:
            return render_template('unauthorized.html',form=form)
    print(x)
    return render_template('index.html',form=form)

# @app.route('/visualisation')
# def visualisation():
#     df = pd.read_csv(csvtarget)
#     pa = df.iloc[:,[1,16,17,8]]

#     d = [dict([(colname, row[i]) for i,colname in enumerate(pa.columns)])for row in pa.values]
    
#     x = json.dumps(d)
#     # po = json.loads(x)
#     records = pa.to_json(orient='index')
#     print(type(records))
#     return render_template('visualisation.html',data=records)

@app.route('/aboutus')
def aboutus():
    return render_template('About_Us.html')

@app.route('/module')
def module():
    return render_template('modules.html')

@app.route('/visualisation')
def visualisation():
    return render_template('visualisation.html')

@app.route('/formforward')
def formforward():
    return render_template('formforward.html')

if __name__ == '__main__':
    app.run(debug=True)