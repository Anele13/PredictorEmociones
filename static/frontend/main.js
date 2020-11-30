let model;

const modelURL = 'http://localhost:5000/model';

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');

const colores = {0:`border border-danger`,
                1:`border border-secondary`,
                2:`border border-success`,
                3:`border border-warning`,
                4:`border border-info`,
                5:`border border-light`,
                6:`border border-dark`,
                7:`border border-white`
}


const predict = async (modelURL) => {
    $('#banner-error').hide()
    $('#preview').hide()
    $('#resultado').hide()
    $('#spinner').show()
    $('#predict').prop("disabled", true);
    $('#predict').html(`<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Trabajando...`)
    preview.innerHTML = " "
    if (!model) model = await tf.loadLayersModel(modelURL);
    const files = fileInput.files;

    [...files].map(async (img, index) => {
        const data = new FormData();
        data.append('file', img);
        data.append('color',colores[index])
        $($('.image-block')[index]).addClass(colores[index])

        const result = await fetch("/api/prepare",
            {
                method: 'POST',
                body: data
            }).then(response => {
                return response.json();
            }).then(result => { 
              if (!result.error){
                for (const [key, value] of Object.entries(result))  {
                  class_names =  {0:'Enojo',
                                  1:'Disgusto',
                                  2:'Felicidad',
                                  3:'Neutral',
                                  4:'Tristeza',
                                  5:'Sorpresa'}
                    const processedImage = tf.tensor2d(value[0])
                    const prediction = model.predict(tf.reshape(processedImage, shape = [1, processedImage.shape[0],processedImage.shape[0], 1]));
                    const label = prediction.argMax(axis = 1).dataSync()[0];                      
                    const preds = prediction.dataSync();
                    const confiabilidad = Math.max( ...preds ) * 100
                    source = "data:image/png;base64," + value[1]
                    color = value[2]
                    $('#spinner').hide()
                    $('#preview').show()
                    $('#resultado').show()
                    $('#predict').prop("disabled", false);
                    $('#predict').html(`Predecir`)
                    renderImageLabel(img, class_names[label], confiabilidad, source, color);
              }
            }
            }).catch(error => {
              $('#banner-error').show()
              $('#spinner').hide();
              $('#predict').prop("disabled", false);
              $('#predict').html(`Predecir`);
            })
    })
};


 

const renderImageLabel = (img, label, confiabilidad, data, color) => {
    const reader = new FileReader();
    reader.onload = () => {
        //$('#img_original').attr('src',reader.result)
        preview.innerHTML += `
                            <div class="card bg-light mb-3 ${color} image-block" style="max-width: 18rem;">
                              <div class="card-header"><strong>${label} </strong></div>
                              <div class="card-body">
                                <div class="image-block">
                                  <img src="${data}" id="source"  style="width: 200px; height: 220px;"/>
                                  <p class="card-text">${confiabilidad}% de confiabilidad </p>
                                </div>
                              </div>
                            </div>`;


                              
    };
    reader.readAsDataURL(img);
};

$('input[type="file"]').change(function() {
    $('.thumbnail').html('');
    $.each(this.files, function() {
      readURL(this);
    })
  });
  
  function readURL(file) {
    var reader = new FileReader();
    reader.onload = function(e) {
      $('.thumbnail').append('<img class=image-block src=' + e.target.result + ' style="width: 200px; height: 220px;"/>');
    }
  
    reader.readAsDataURL(file);
  }



predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => {preview.innerHTML = ""; $('#preview').hide(); $('#resultado').hide()});
